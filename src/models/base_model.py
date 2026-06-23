"""
src/models/base_model.py

Shared interface that all three models (XGBoost, Dixon-Coles, Agent)
implement, enabling plug-and-play comparison and Monte Carlo simulation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class MatchPrediction:
    """
    Standard output for every prediction model.

    Probabilities are from the HOME TEAM's perspective:
        p_home_win + p_draw + p_away_win == 1.0

    lambda_home / lambda_away: expected goals (optional, used for
    Poisson-based score simulation in the Monte Carlo layer).
    """
    p_home_win: float
    p_draw:     float
    p_away_win: float
    lambda_home: Optional[float] = None
    lambda_away: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        total = self.p_home_win + self.p_draw + self.p_away_win
        # Normalize in case of floating-point drift
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Probabilities don't sum to 1: {total:.6f}")
        self.p_home_win /= total
        self.p_draw     /= total
        self.p_away_win /= total

    def as_dict(self) -> dict:
        return {
            "p_home_win":  self.p_home_win,
            "p_draw":      self.p_draw,
            "p_away_win":  self.p_away_win,
            "lambda_home": self.lambda_home,
            "lambda_away": self.lambda_away,
            **self.metadata,
        }

    def most_likely(self) -> str:
        options = {
            "home_win": self.p_home_win,
            "draw":     self.p_draw,
            "away_win": self.p_away_win,
        }
        return max(options, key=options.get)

    def __repr__(self) -> str:
        return (
            f"MatchPrediction("
            f"W={self.p_home_win:.3f}, "
            f"D={self.p_draw:.3f}, "
            f"L={self.p_away_win:.3f}"
            + (f", xG={self.lambda_home:.2f}-{self.lambda_away:.2f}" if self.lambda_home else "")
            + ")"
        )


class BaseMatchPredictor(ABC):
    """Abstract interface shared by all three models."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train on historical match data (with feature columns already built)."""
        ...

    @abstractmethod
    def predict(
        self,
        home_team: str,
        away_team: str,
        features: Optional[dict] = None,
        **kwargs,
    ) -> MatchPrediction:
        """Return outcome probabilities for a single match."""
        ...

    def predict_batch(self, matches: list[dict]) -> list[MatchPrediction]:
        return [self.predict(**m) for m in matches]

    def simulate_score_distribution(
        self,
        home_team: str,
        away_team: str,
        features: Optional[dict] = None,
        n_sim: int = 50_000,
    ) -> dict:
        """
        Monte Carlo score distribution using Poisson sampling.
        Only meaningful if the model provides lambda estimates.
        """
        pred = self.predict(home_team, away_team, features)

        if pred.lambda_home is None or pred.lambda_away is None:
            return {"prediction": pred, "scores": None}

        gh = np.random.poisson(pred.lambda_home, n_sim)
        ga = np.random.poisson(pred.lambda_away, n_sim)

        score_counts: dict[str, int] = {}
        for h, a in zip(gh, ga):
            k = f"{h}-{a}"
            score_counts[k] = score_counts.get(k, 0) + 1

        top = sorted(score_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            "prediction":  pred,
            "top_scores":  [(s, c / n_sim) for s, c in top],
            "avg_xg_home": float(np.mean(gh)),
            "avg_xg_away": float(np.mean(ga)),
        }
