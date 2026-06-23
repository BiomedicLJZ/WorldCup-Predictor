"""
src/models/agent_model.py

Deep LangGraph agent combining both statistical models with LLM reasoning.

Architecture (v2 — parallel nodes + confidence routing):

    START → [get_xgb ∥ get_poisson ∥ search_news]   (all three in parallel)
         → aggregate  (weighted ensemble + disagreement score)
         → [router: disagreement < threshold AND no impact news?]
               ↙ fast path              ↘ deep path
           calibrate              synthesize (LLM) → calibrate
               ↓                                          ↓
              END ←──────────────────────────────────── END

The LLM is only invoked when it can add genuine value:
  • Models disagree beyond confidence_threshold (default 8%)
  • OR news contains impact keywords (injuries, suspensions, coaching change)

This avoids LLM cost on clear-cut matches while using AI judgment precisely
where statistics are uncertain or contextual information matters.
"""

from __future__ import annotations
import json
import logging
from typing import Optional, TypedDict
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

try:
    from langgraph.graph import StateGraph, END
    try:
        from langgraph.graph import START
    except ImportError:
        START = "__start__"   # LangGraph < 0.2 fallback (START == "__start__" internally)
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_nvidia_ai_endpoints import ChatNvidia
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    START = None

try:
    from langchain_community.tools import DuckDuckGoSearchRun
    _SEARCH_AVAILABLE = True
except ImportError:
    _SEARCH_AVAILABLE = False

import pandas as pd
from .base_model import BaseMatchPredictor, MatchPrediction
from .xgboost_model import XGBoostPredictor
from .poisson_model import DixonColesPredictor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────

# News keywords whose presence triggers LLM even when models agree
_NEWS_IMPACT_KEYWORDS = [
    "injured", "suspend", "absent", "withdraw", "ruled out",
    "doubt", "crisis", "sacked", "fired", "ill", "emergency",
]

# Neutral-venue WC prior (empirical, 1966-2022 WC matches)
_WC_NEUTRAL_PRIOR = {"p_home_win": 0.38, "p_draw": 0.24, "p_away_win": 0.38}

# Shrinkage weight toward prior; corrects systematic draw under-prediction in ML models
_CALIBRATION_ALPHA = 0.08


# ─────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────

class MatchPredictionState(TypedDict):
    home_team:            str
    away_team:            str
    is_neutral:           bool
    match_context:        str
    features:             Optional[dict]

    # Written by parallel nodes (each writes its own key — no reducer needed)
    xgb_prediction:       Optional[dict]
    poisson_prediction:   Optional[dict]
    news_context:         Optional[str]

    # Written by aggregate node
    statistical_ensemble: Optional[dict]   # 40% XGB + 60% DC weighted average
    disagreement_score:   float            # L1 distance in [0, 1]

    # Written by synthesize / calibrate
    final_prediction:     Optional[dict]
    reasoning:            Optional[str]


# ─────────────────────────────────────────────────────────
# AGENT PREDICTOR
# ─────────────────────────────────────────────────────────

class AgentPredictor(BaseMatchPredictor):
    """
    Deep LangGraph workflow: XGBoost + Dixon-Coles + optional LLM synthesis.

    Improvements over v1:
      - All three data-gathering nodes (XGB, Poisson, search) run IN PARALLEL
      - Confidence router skips LLM when models agree → faster + cheaper
      - Calibration node corrects systematic draw under-prediction
      - Richer LLM prompt includes disagreement score and clear adjustment rules
      - Uniform graph topology (search node always present, fast-returns if disabled)
    """

    @property
    def name(self) -> str:
        return "Agent (XGB + Dixon-Coles + LLM)"

    def __init__(
        self,
        xgb_model:             XGBoostPredictor,
        poisson_model:         DixonColesPredictor,
        llm_config:            dict,
        enable_search:         bool = True,
        confidence_threshold:  float = 0.08,
    ):
        """
        Args:
            xgb_model:            Already-trained XGBoostPredictor.
            poisson_model:        Already-trained DixonColesPredictor.
            llm_config:           dict with keys: model, api_key, base_url.
            enable_search:        If True, query DuckDuckGo for team news.
            confidence_threshold: L1 disagreement above which LLM is invoked.
                                  0.08 = 8% divergence between models triggers LLM.
        """
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph not installed. Run: pip install langgraph langchain-openai"
            )

        self.xgb    = xgb_model
        self.poisson = poisson_model
        self.enable_search = enable_search and _SEARCH_AVAILABLE
        self.confidence_threshold = confidence_threshold

        self.llm = ChatNvidia(
            model=    llm_config.get("model", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),
            api_key=  llm_config.get(os.getenv("NVIDIA_API_KEY_ENV", "NVIDIA_API_KEY")),
            base_url= llm_config.get("base_url"),
            temperature=0.6,
        )

        self.search_tool = DuckDuckGoSearchRun() if self.enable_search else None
        self.graph = self._build_graph()

    # ─────────────────────────────────────────────────
    # FIT (pass-through — sub-models are pre-trained)
    # ─────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        logger.info(
            "AgentPredictor uses pre-trained sub-models. "
            f"XGBoost: {'✓' if self.xgb.model else '✗'} | "
            f"Dixon-Coles: {'✓' if self.poisson.attack else '✗'}"
        )

    # ─────────────────────────────────────────────────
    # PREDICT — runs the LangGraph workflow
    # ─────────────────────────────────────────────────

    def predict(
        self,
        home_team:     str,
        away_team:     str,
        features:      Optional[dict] = None,
        match_context: str = "FIFA World Cup 2026",
        **kwargs,
    ) -> MatchPrediction:
        initial: MatchPredictionState = {
            "home_team":            home_team,
            "away_team":            away_team,
            "is_neutral":           True,
            "match_context":        match_context,
            "features":             features,
            "xgb_prediction":       None,
            "poisson_prediction":   None,
            "news_context":         None,
            "statistical_ensemble": None,
            "disagreement_score":   0.5,   # conservative default → routes to LLM
            "final_prediction":     None,
            "reasoning":            None,
        }

        final = self.graph.invoke(initial)
        p     = final["final_prediction"]
        route = "fast" if final.get("disagreement_score", 1.0) < self.confidence_threshold else "deep"

        return MatchPrediction(
            p_home_win=  p["p_home_win"],
            p_draw=      p["p_draw"],
            p_away_win=  p["p_away_win"],
            lambda_home= p.get("lambda_home"),
            lambda_away= p.get("lambda_away"),
            metadata={
                "model":        self.name,
                "reasoning":    final.get("reasoning", ""),
                "disagreement": final.get("disagreement_score", 0.0),
                "route":        route,
                "xgb":          final.get("xgb_prediction"),
                "poisson":      final.get("poisson_prediction"),
            },
        )

    # ─────────────────────────────────────────────────
    # GRAPH CONSTRUCTION
    # ─────────────────────────────────────────────────

    def _build_graph(self):
        wf = StateGraph(MatchPredictionState)

        # ── Nodes ──────────────────────────────────────────────────────────
        wf.add_node("get_xgb",     self._node_get_xgb)
        wf.add_node("get_poisson", self._node_get_poisson)
        wf.add_node("search_news", self._node_search_news)  # always present
        wf.add_node("aggregate",   self._node_aggregate)
        wf.add_node("synthesize",  self._node_synthesize)
        wf.add_node("calibrate",   self._node_calibrate)

        # ── Parallel fan-out from START ─────────────────────────────────────
        # LangGraph executes all three concurrently; each writes its own state
        # key so there are no write conflicts between parallel nodes.
        wf.add_edge(START, "get_xgb")
        wf.add_edge(START, "get_poisson")
        wf.add_edge(START, "search_news")

        # ── Fan-in: aggregate waits for all three to complete ──────────────
        wf.add_edge("get_xgb",     "aggregate")
        wf.add_edge("get_poisson", "aggregate")
        wf.add_edge("search_news", "aggregate")

        # ── Confidence router ──────────────────────────────────────────────
        wf.add_conditional_edges(
            "aggregate",
            self._route_by_confidence,
            {"fast": "calibrate", "deep": "synthesize"},
        )

        # ── Both paths end at calibrate → END ──────────────────────────────
        wf.add_edge("synthesize", "calibrate")
        wf.add_edge("calibrate",  END)

        return wf.compile()

    # ─────────────────────────────────────────────────
    # NODES
    # ─────────────────────────────────────────────────

    def _node_get_xgb(self, state: MatchPredictionState) -> dict:
        try:
            pred = self.xgb.predict(
                state["home_team"], state["away_team"], state.get("features")
            )
            return {"xgb_prediction": pred.as_dict()}
        except Exception as e:
            logger.error(f"XGBoost node failed: {e}")
            return {"xgb_prediction": self._uniform()}

    def _node_get_poisson(self, state: MatchPredictionState) -> dict:
        try:
            pred = self.poisson.predict(
                state["home_team"], state["away_team"],
                is_neutral=state.get("is_neutral", True),
            )
            return {"poisson_prediction": pred.as_dict()}
        except Exception as e:
            logger.error(f"Poisson node failed: {e}")
            return {"poisson_prediction": self._uniform()}

    def _node_search_news(self, state: MatchPredictionState) -> dict:
        if not self.enable_search or not self.search_tool:
            return {"news_context": "Search not enabled."}
        home, away = state["home_team"], state["away_team"]
        try:
            q1  = f"{home} vs {away} World Cup 2026 squad injuries suspensions form"
            q2  = f"{home} national team 2026 latest news"
            q3  = f"{away} national team 2026 latest news"
            ctx = (
                self.search_tool.run(q1)[:500] + "\n\n" +
                self.search_tool.run(q2)[:300] + "\n\n" +
                self.search_tool.run(q3)[:300]
            )
        except Exception as e:
            logger.warning(f"Search node failed: {e}")
            ctx = "No recent news available."
        return {"news_context": ctx}

    def _node_aggregate(self, state: MatchPredictionState) -> dict:
        """Weighted ensemble + L1 disagreement score."""
        xgb = state.get("xgb_prediction")    or self._uniform()
        poi = state.get("poisson_prediction") or self._uniform()

        # Dixon-Coles is more reliable for neutral-venue tournament matches
        w_xgb, w_poi = 0.40, 0.60
        ens = {
            "p_home_win": w_xgb * xgb["p_home_win"] + w_poi * poi["p_home_win"],
            "p_draw":     w_xgb * xgb["p_draw"]     + w_poi * poi["p_draw"],
            "p_away_win": w_xgb * xgb["p_away_win"] + w_poi * poi["p_away_win"],
        }
        total = sum(ens.values())
        ens   = {k: v / total for k, v in ens.items()}

        # L1 disagreement normalised to [0, 1]
        disagreement = (
            abs(xgb["p_home_win"] - poi["p_home_win"]) +
            abs(xgb["p_draw"]     - poi["p_draw"])     +
            abs(xgb["p_away_win"] - poi["p_away_win"])
        ) / 2.0

        ens["lambda_home"] = (
            (xgb.get("lambda_home") or 1.3) + (poi.get("lambda_home") or 1.3)
        ) / 2.0
        ens["lambda_away"] = (
            (xgb.get("lambda_away") or 1.1) + (poi.get("lambda_away") or 1.1)
        ) / 2.0

        return {
            "statistical_ensemble": ens,
            "disagreement_score":   float(disagreement),
        }

    def _route_by_confidence(self, state: MatchPredictionState) -> str:
        """Return 'fast' (skip LLM) or 'deep' (invoke LLM synthesis)."""
        score = state.get("disagreement_score", 1.0)
        news  = (state.get("news_context") or "").lower()

        if score >= self.confidence_threshold:
            return "deep"
        if any(kw in news for kw in _NEWS_IMPACT_KEYWORDS):
            return "deep"
        return "fast"

    def _node_synthesize(self, state: MatchPredictionState) -> dict:
        home = state["home_team"]
        away = state["away_team"]
        xgb  = state.get("xgb_prediction")    or self._uniform()
        poi  = state.get("poisson_prediction") or self._uniform()
        ens  = state.get("statistical_ensemble") or self._weighted_ensemble(xgb, poi)
        news = state.get("news_context", "Not available")
        ctx  = state.get("match_context", "FIFA World Cup")
        disg = state.get("disagreement_score", 0.0)

        if disg < 0.05:
            disg_label = "LOW — models closely agree"
        elif disg < 0.15:
            disg_label = "MODERATE — some divergence between models"
        else:
            disg_label = "HIGH — models diverge significantly; investigate carefully"

        system_prompt = (
            "You are an elite football analyst combining statistical modeling with tactical intelligence.\n\n"
            "CORE PRINCIPLE: Statistical models are your anchor. Deviate ONLY with concrete, specific evidence.\n\n"
            "ADJUSTMENT RULES:\n"
            "  Disagreement LOW (<0.05): Stay within 0.03 of statistical consensus.\n"
            "  Disagreement MODERATE (0.05-0.15): Light adjustment based on strong evidence.\n"
            "  Disagreement HIGH (>0.15): Determine which model better fits this context:\n"
            "    • XGBoost excels at: recent form, momentum, historical H2H patterns\n"
            "    • Dixon-Coles excels at: team attack/defense rates, neutral-venue tournaments\n\n"
            "VALID ADJUSTMENT TRIGGERS (only these justify deviating from stats):\n"
            "  ✓ Confirmed absence of a starting-11 player (not just a squad player)\n"
            "  ✓ Coaching change in the past 30 days\n"
            "  ✓ Documented internal team crisis or disciplinary incident\n"
            "  ✗ Vague 'team is on good form' — models already capture this\n"
            "  ✗ Historical tournament reputation alone\n\n"
            "RESPONSE: Valid JSON only — no prose outside the JSON block.\n"
            "{\n"
            '  "p_home_win": 0.XX,\n'
            '  "p_draw": 0.XX,\n'
            '  "p_away_win": 0.XX,\n'
            '  "lambda_home": X.XX,\n'
            '  "lambda_away": X.XX,\n'
            '  "reasoning": "1-2 specific, evidence-backed sentences",\n'
            '  "confidence": "low|medium|high",\n'
            '  "key_factor": "Single most important factor in your assessment"\n'
            "}\n"
            "Probabilities must sum to exactly 1.0."
        )

        user_prompt = (
            f"Match: {home} vs {away}\n"
            f"Context: {ctx} (neutral venue)\n\n"
            f"━━ MODEL PREDICTIONS ━━\n\n"
            f"XGBoost (form + Elo + H2H + momentum):\n"
            f"  {home}: {xgb.get('p_home_win', 0):.1%}  │  "
            f"Draw: {xgb.get('p_draw', 0):.1%}  │  "
            f"{away}: {xgb.get('p_away_win', 0):.1%}\n\n"
            f"Dixon-Coles (Bivariate Poisson — attack/defense parameters):\n"
            f"  {home}: {poi.get('p_home_win', 0):.1%}  │  "
            f"Draw: {poi.get('p_draw', 0):.1%}  │  "
            f"{away}: {poi.get('p_away_win', 0):.1%}\n"
            f"  Expected goals: {home} {poi.get('lambda_home') or 1.3:.2f} — "
            f"{poi.get('lambda_away') or 1.1:.2f} {away}\n\n"
            f"Weighted Consensus (40% XGB / 60% DC):\n"
            f"  {home}: {ens.get('p_home_win', 0):.1%}  │  "
            f"Draw: {ens.get('p_draw', 0):.1%}  │  "
            f"{away}: {ens.get('p_away_win', 0):.1%}\n\n"
            f"Model Disagreement: {disg:.3f}  ({disg_label})\n\n"
            f"━━ CONTEXTUAL INTELLIGENCE ━━\n"
            f"{news[:1000] if news else 'No news available — rely on statistical models.'}\n\n"
            f"Provide your expert synthesis."
        )

        try:
            resp = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            text = resp.content.strip()

            for fence in ["```json", "```"]:
                if fence in text:
                    text = text.split(fence)[1].split("```")[0].strip()
                    break

            result = json.loads(text)

            total = result["p_home_win"] + result["p_draw"] + result["p_away_win"]
            result["p_home_win"] /= total
            result["p_draw"]     /= total
            result["p_away_win"] /= total

            # Fall back to ensemble lambdas if LLM omitted them
            result.setdefault("lambda_home", ens.get("lambda_home", 1.3))
            result.setdefault("lambda_away", ens.get("lambda_away", 1.1))

            return {
                "final_prediction": result,
                "reasoning":        result.get("reasoning", ""),
            }

        except Exception as e:
            logger.error(f"LLM synthesis failed ({e}). Falling back to ensemble.")
            return {
                "final_prediction": ens,
                "reasoning":        "Weighted ensemble (LLM unavailable).",
            }

    def _node_calibrate(self, state: MatchPredictionState) -> dict:
        """
        Bayesian shrinkage toward neutral-venue WC priors.
        Corrects systematic draw under-prediction common in ML models.
        Runs on BOTH the fast path (no LLM) and after LLM synthesis.
        """
        pred = (
            state.get("final_prediction")     or
            state.get("statistical_ensemble") or
            self._uniform()
        )

        calibrated = {
            k: (1 - _CALIBRATION_ALPHA) * pred.get(k, 1 / 3)
               + _CALIBRATION_ALPHA * _WC_NEUTRAL_PRIOR[k]
            for k in ["p_home_win", "p_draw", "p_away_win"]
        }
        total = sum(calibrated.values())
        calibrated = {k: v / total for k, v in calibrated.items()}

        return {
            "final_prediction": {
                **calibrated,
                "lambda_home": pred.get("lambda_home", 1.3),
                "lambda_away": pred.get("lambda_away", 1.1),
                "reasoning":   pred.get("reasoning", "Statistical ensemble with WC calibration."),
                "confidence":  pred.get("confidence", "medium"),
                "key_factor":  pred.get("key_factor", ""),
            }
        }

    # ─────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────

    @staticmethod
    def _uniform() -> dict:
        return {"p_home_win": 1 / 3, "p_draw": 1 / 3, "p_away_win": 1 / 3}

    @staticmethod
    def _weighted_ensemble(xgb: dict, poi: dict, w_xgb: float = 0.40) -> dict:
        w_poi = 1.0 - w_xgb
        pw  = w_xgb * xgb.get("p_home_win", 1 / 3) + w_poi * poi.get("p_home_win", 1 / 3)
        pd_ = w_xgb * xgb.get("p_draw",     1 / 3) + w_poi * poi.get("p_draw",     1 / 3)
        pl  = w_xgb * xgb.get("p_away_win", 1 / 3) + w_poi * poi.get("p_away_win", 1 / 3)
        total = pw + pd_ + pl
        return {
            "p_home_win":  pw  / total,
            "p_draw":      pd_ / total,
            "p_away_win":  pl  / total,
            "lambda_home": ((xgb.get("lambda_home") or 1.3) + (poi.get("lambda_home") or 1.3)) / 2,
            "lambda_away": ((xgb.get("lambda_away") or 1.1) + (poi.get("lambda_away") or 1.1)) / 2,
        }

    @staticmethod
    def _ensemble_avg(xgb: dict, poi: dict) -> dict:
        """Backward-compat alias (equal weighting)."""
        return AgentPredictor._weighted_ensemble(xgb, poi, w_xgb=0.50)
