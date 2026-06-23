"""
main.py — World Cup 2026 Prediction Pipeline

Orchestrates:
    1. Data download + loading
    2. Elo computation
    3. Feature engineering
    4. Model training (XGBoost + Dixon-Coles + optional Agent)
    5. Backtesting + comparison table + comparison plot
    6. Monte Carlo simulation of WC 2026

Run:
    python main.py                  # full pipeline
    python main.py --skip-backtest  # skip evaluation, go straight to simulation
    python main.py --match "Brazil" "France"  # predict a single match
"""

from __future__ import annotations
import argparse
import logging
import os
import yaml
from pathlib import Path

import pandas as pd

from src.data.loader import DataLoader
from src.features.elo import EloSystem
from src.features.engineering import FeatureEngineer
from src.models.xgboost_model import XGBoostPredictor
from src.models.poisson_model import DixonColesPredictor
from src.simulation.monte_carlo import WorldCupSimulator, WC_2026_GROUPS, fetch_live_wc_results
from src.evaluation.metrics import ModelEvaluator

# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("WC2026")
# ─────────────────────────────────────────────────────────


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════

def run_pipeline(config: dict, skip_backtest: bool = False) -> dict:
    output_dir = Path(config.get("output", {}).get("dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    banner("WORLD CUP 2026 PREDICTION SYSTEM")

    # ── 1. DATA ──────────────────────────────────────────
    section("1 / 5  DATA")
    loader = DataLoader(data_dir=config["data"]["raw_dir"])
    loader.download_data(force=config["data"].get("force_download", False))

    full_df = loader.load_results(
        start_year=config["data"]["start_year"],
        include_friendlies=config["data"]["include_friendlies"],
    )

    # ── 2. ELO ───────────────────────────────────────────
    section("2 / 5  ELO RATINGS")
    elo = EloSystem(
        k_base=config["elo"]["k_base"],
        home_advantage=config["elo"]["home_advantage"],
    )
    full_df = elo.compute_all(full_df)

    top10 = elo.get_top_teams(10)
    logger.info("Current top-10 Elo rankings:")
    print(top10.to_string(index=False))

    # ── 3. FEATURES ──────────────────────────────────────
    section("3 / 5  FEATURE ENGINEERING")
    fe = FeatureEngineer(elo, form_window=config["features"]["form_window"])
    full_df = fe.build_features(full_df)

    # Train / test split
    cutoff = config["evaluation"]["train_cutoff_year"]
    train_df = full_df[full_df["date"].dt.year < cutoff].copy()
    test_df  = full_df[
        (full_df["date"].dt.year >= cutoff) &
        (full_df["tournament"] == "FIFA World Cup")
    ].copy()

    logger.info(f"Train set : {len(train_df):,} matches  (< {cutoff})")
    logger.info(f"Test set  : {len(test_df):,} World Cup matches (>= {cutoff})")

    # ── 4. MODELS ────────────────────────────────────────
    section("4 / 5  MODELS")
    xgb_model = XGBoostPredictor(
        feature_engineer=fe,
        n_optuna_trials=config["xgboost"]["n_optuna_trials"],
        n_cv_splits=config["xgboost"]["n_cv_splits"],
    )
    dc_model = DixonColesPredictor(
        xi=config["dixon_coles"]["xi"],
        max_goals=config["dixon_coles"]["max_goals"],
    )

    models = {
        "XGBoost":      xgb_model,
        "Dixon-Coles":  dc_model,
    }

    # Optional Agent (requires LangGraph + LLM API key)
    if config.get("agent", {}).get("enabled", False):
        _setup_agent(models, xgb_model, dc_model, train_df, fe, config)

    # ── 5. EVALUATION ────────────────────────────────────
    if not skip_backtest and len(test_df) > 0:
        section("5 / 5  BACKTESTING + COMPARISON")
        evaluator = ModelEvaluator(models)
        evaluator.backtest(
            train_df=train_df,
            test_df=test_df,
            feature_engineer=fe,
        )
        comparison_df = evaluator.compare()

        plot_path = config.get("output", {}).get("comparison_plot", "output/comparison.png")
        evaluator.plot_comparison(save_path=plot_path)

        comparison_df.to_csv(output_dir / "comparison.csv", index=False)
    else:
        # If skipping backtest, still train models for simulation
        logger.info("Training models on full dataset (backtest skipped)…")
        for name, model in models.items():
            logger.info(f"  Training {name}…")
            model.fit(train_df)

    # ── 6. SIMULATE WC 2026 ──────────────────────────────
    section("MONTE CARLO — FIFA WORLD CUP 2026")

    # Optionally refresh known results from ESPN (live tournament data)
    known_results = None
    if config["simulation"].get("fetch_live_results", False):
        logger.info("Fetching live WC 2026 results from ESPN…")
        known_results = fetch_live_wc_results()
        n_live = len(known_results)
        logger.info(f"  {n_live} matches locked in for simulation.")

    all_sim_results = {}
    for model_name, model in models.items():
        logger.info(f"\nSimulating with {model_name}…")
        sim = WorldCupSimulator(
            predictor=model,
            groups=WC_2026_GROUPS,
            n_simulations=config["simulation"]["n_simulations"],
            known_results=known_results,
        )
        probs = sim.champion_probabilities()
        all_sim_results[model_name] = probs

        logger.info(f"\n🏆 {model_name} — Top 10 champions:")
        print(
            probs.head(10)[["team", "p_champion", "p_finalist", "p_semi"]]
            .rename(columns={
                "p_champion": "Champion",
                "p_finalist": "Finalist",
                "p_semi":     "Semi-final",
            })
            .to_string(index=False)
        )

        out_file = output_dir / f"sim_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
        probs.to_csv(out_file, index=False)
        logger.info(f"Saved → {out_file}")

    # Combined comparison table (champion probabilities from all models)
    _print_combined_champion_table(all_sim_results)

    banner("PIPELINE COMPLETE ✓")
    return models


# ══════════════════════════════════════════════════════════
#  SINGLE MATCH PREDICTION
# ══════════════════════════════════════════════════════════

def predict_match(
    home_team: str,
    away_team: str,
    config: dict,
) -> None:
    """
    Train both models and print a side-by-side prediction for one match.
    Useful for quick ad-hoc queries during the tournament.
    """
    banner(f"PREDICTION:  {home_team}  vs  {away_team}")

    loader = DataLoader(data_dir=config["data"]["raw_dir"])
    loader.download_data()
    full_df = loader.load_results(start_year=config["data"]["start_year"])

    elo = EloSystem(**{k: config["elo"][k] for k in ["k_base", "home_advantage"]})
    full_df = elo.compute_all(full_df)

    fe = FeatureEngineer(elo, form_window=config["features"]["form_window"])
    full_df = fe.build_features(full_df)

    xgb = XGBoostPredictor(fe, n_optuna_trials=20)
    dc  = DixonColesPredictor(
        xi=config["dixon_coles"]["xi"],
        max_goals=config["dixon_coles"]["max_goals"],
    )

    for name, model in [("XGBoost", xgb), ("Dixon-Coles", dc)]:
        logger.info(f"Training {name}…")
        model.fit(full_df)

    features = fe.build_single_match_features(home_team, away_team)

    xgb_pred = xgb.predict(home_team, away_team, features)
    dc_pred  = dc.predict(home_team, away_team, features)

    print(f"\n{'-'*55}")
    print(f"  {home_team:20s}  vs  {away_team}")
    print(f"{'-'*55}")
    print(f"  {'':22s}  {'XGBoost':>10s}  {'Dixon-Coles':>12s}")
    print(f"  {'':22s}  {'-'*10}  {'-'*12}")
    print(f"  {home_team + ' win':22s}  {xgb_pred.p_home_win*100:>9.1f}%  {dc_pred.p_home_win*100:>11.1f}%")
    print(f"  {'Draw':22s}  {xgb_pred.p_draw*100:>9.1f}%  {dc_pred.p_draw*100:>11.1f}%")
    print(f"  {away_team + ' win':22s}  {xgb_pred.p_away_win*100:>9.1f}%  {dc_pred.p_away_win*100:>11.1f}%")
    if dc_pred.lambda_home:
        print(f"\n  Expected Goals (DC)   {dc_pred.lambda_home:.2f}  -  {dc_pred.lambda_away:.2f}")
    print(f"{'-'*55}")

    # Score distribution
    dist = dc.simulate_score_distribution(home_team, away_team, features)
    print(f"\n  Most likely scores (Dixon-Coles):")
    for score, prob in (dist.get("top_scores") or [])[:5]:
        print(f"    {score:>5s}   {prob*100:.1f}%")
    print()


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

def _setup_agent(models, xgb_model, dc_model, train_df, fe, config):
    """Lazy-import and configure the LangGraph agent."""
    try:
        from src.models.agent_model import AgentPredictor

        llm_cfg = config["agent"]["llm"].copy()
        # Expand env variables in api_key
        api_key = llm_cfg.get("api_key", "")
        if api_key and api_key.startswith("${"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "")
            llm_cfg["api_key"] = api_key

        if not api_key:
            logger.warning("Agent enabled in config but no API key found. Skipping.")
            return

        # Train sub-models first if not already trained
        if xgb_model.model is None:
            xgb_model.fit(train_df)
        if not dc_model.attack:
            dc_model.fit(train_df)

        agent = AgentPredictor(
            xgb_model=xgb_model,
            poisson_model=dc_model,
            llm_config=llm_cfg,
            enable_search=config["agent"].get("enable_search", True),
            confidence_threshold=config["agent"].get("confidence_threshold", 0.08),
        )
        models["Agent (XGB + DC + LLM)"] = agent
        logger.info("Agent model added to pipeline.")

    except ImportError as e:
        logger.warning(f"Could not load AgentPredictor: {e}")


def _print_combined_champion_table(all_results: dict) -> None:
    """Side-by-side champion probability table across all models."""
    if len(all_results) < 2:
        return

    dfs = []
    for model_name, df in all_results.items():
        short = model_name.split()[0]  # XGBoost -> XGB, Dixon-Coles -> Dixon
        col = df[["team", "p_champion"]].rename(columns={"p_champion": f"P(Win) [{short}]"})
        dfs.append(col.set_index("team"))

    combined = pd.concat(dfs, axis=1).fillna(0.0)
    combined = combined.sort_values(combined.columns[0], ascending=False).head(16)

    print("\n" + "="*55)
    print("  COMBINED CHAMPION PROBABILITIES (top 16)")
    print("="*55)
    for col in combined.columns:
        combined[col] = combined[col].map(lambda x: f"{x*100:.1f}%")
    print(combined.to_string())
    print("="*55 + "\n")


def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def section(msg: str) -> None:
    print(f"\n{'-'*60}")
    logger.info(msg)
    print(f"{'-'*60}")


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Cup 2026 Predictor")
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--match",         nargs=2, metavar=("HOME", "AWAY"),
                        help='Quick single-match prediction, e.g. --match "Brazil" "France"')
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.match:
        predict_match(args.match[0], args.match[1], cfg)
    else:
        run_pipeline(cfg, skip_backtest=args.skip_backtest)
