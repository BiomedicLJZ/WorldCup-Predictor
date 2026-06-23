"""
src/evaluation/metrics.py

Evaluation and comparison framework for all three match prediction models.

Metrics used:
    RPS  (Ranked Probability Score)  — primary metric
        Ideal for ordinal 3-class outcomes (Win > Draw > Loss).
        Penalizes predictions that are wrong AND far from the truth.
        RPS = 0 is perfect. Baseline (uniform priors) ≈ 0.333.

    Log-Loss
        Standard probabilistic loss. Penalizes confident wrong predictions
        exponentially. Sensitive to calibration.

    Accuracy
        Simple percentage of correct result classifications.
        Ignores probability quality; useful as sanity check.

    Brier Score
        Mean squared error on probabilities per outcome class.
        Good for calibration analysis.

    Calibration Curve
        Plots "predicted probability" vs "actual fraction of positives".
        A well-calibrated model lies on the diagonal.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from typing import Optional
import logging

from ..models.base_model import BaseMatchPredictor
from ..features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates and compares prediction models via temporal backtesting.

    Temporal integrity: each model is trained on data up to year Y,
    then evaluated on matches from year Y onward — no future leakage.
    """

    def __init__(self, models: dict[str, BaseMatchPredictor]):
        """
        Args:
            models: {"XGBoost": xgb_model, "Dixon-Coles": dc_model, ...}
        """
        self.models = models
        self.results_: dict[str, dict] = {}

    # ─────────────────────────────────────────────────
    # BACKTESTING
    # ─────────────────────────────────────────────────

    def backtest(
        self,
        train_df: pd.DataFrame,
        test_df:  pd.DataFrame,
        feature_engineer: FeatureEngineer,
    ) -> dict:
        """
        Train each model on train_df, evaluate on test_df.

        Recommended splits:
            train_df = all matches where date.year < 2018
            test_df  = FIFA World Cup 2018 matches only

        Args:
            train_df: Feature-enriched training DataFrame
            test_df:  Raw test DataFrame (features computed at prediction time)
            feature_engineer: Shared FeatureEngineer (Elo already computed)
        """
        for model_name, model in self.models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating: {model_name}")
            logger.info(f"{'='*50}")

            # Train
            logger.info("Training…")
            model.fit(train_df)

            # Predict on test matches
            preds, actuals = [], []
            for _, row in test_df.iterrows():
                try:
                    features = feature_engineer.build_single_match_features(
                        home_team=row["home_team"],
                        away_team=row["away_team"],
                        is_neutral=bool(row.get("neutral", True)),
                        reference_date=row["date"],
                    )
                    pred = model.predict(row["home_team"], row["away_team"], features)
                    preds.append([pred.p_home_win, pred.p_draw, pred.p_away_win])
                    actuals.append(int(row["result_numeric"]))

                except Exception as e:
                    logger.warning(
                        f"Skipping {row['home_team']} vs {row['away_team']}: {e}"
                    )

            if not preds:
                logger.warning(f"No predictions generated for {model_name}!")
                continue

            P = np.array(preds)
            Y = np.array(actuals)

            metrics = self._compute_all_metrics(P, Y)
            self.results_[model_name] = {
                "metrics":     metrics,
                "predictions": P,
                "actuals":     Y,
                "n_matches":   len(Y),
            }

            logger.info(f"  n_matches : {len(Y)}")
            logger.info(f"  RPS       : {metrics['rps']:.4f}")
            logger.info(f"  Log-Loss  : {metrics['log_loss']:.4f}")
            logger.info(f"  Accuracy  : {metrics['accuracy']:.3f}")

        return self.results_

    # ─────────────────────────────────────────────────
    # COMPARISON TABLE
    # ─────────────────────────────────────────────────

    def compare(self) -> pd.DataFrame:
        """Return a formatted comparison DataFrame."""
        if not self.results_:
            raise RuntimeError("Call backtest() first.")

        rows = []
        for name, res in self.results_.items():
            m = res["metrics"]
            rows.append({
                "Model":          name,
                "RPS ↓":          round(m["rps"],        4),
                "Log-Loss ↓":     round(m["log_loss"],   4),
                "Accuracy ↑":     round(m["accuracy"],   3),
                "Brier (avg) ↓":  round(m["brier_avg"],  4),
                "n":              res["n_matches"],
            })

        df = pd.DataFrame(rows).sort_values("RPS ↓")
        print("\n" + "="*65)
        print("MODEL COMPARISON RESULTS")
        print("="*65)
        print(df.to_string(index=False))
        print("="*65)
        return df

    # ─────────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────────

    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """
        4-panel comparison plot:
            1. RPS (bar)
            2. Log-Loss (bar)
            3. Calibration curve (Win probability)
            4. Per-class accuracy (grouped bar)
        """
        if not self.results_:
            raise RuntimeError("Call backtest() first.")

        model_names = list(self.results_.keys())
        palette = ["#4285F4", "#34A853", "#FF6D00", "#EA4335"][:len(model_names)]

        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(
            "World Cup Prediction — Model Comparison",
            fontsize=15, fontweight="bold", y=1.01
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)
        axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

        # ── 1. RPS ───────────────────────────────────
        ax = axes[0]
        rps_vals = [self.results_[m]["metrics"]["rps"] for m in model_names]
        bars = ax.bar(model_names, rps_vals, color=palette, alpha=0.85, edgecolor="white")
        ax.set_title("Ranked Probability Score  (lower = better)", fontweight="bold")
        ax.set_ylabel("RPS")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_ylim(0, max(rps_vals) * 1.25)
        ax.tick_params(axis="x", rotation=15)
        # Baseline: uniform prediction
        ax.axhline(1/3, color="gray", linestyle="--", linewidth=1, label="Uniform baseline")
        ax.legend(fontsize=8)

        # ── 2. Log-Loss ──────────────────────────────
        ax = axes[1]
        ll_vals = [self.results_[m]["metrics"]["log_loss"] for m in model_names]
        bars = ax.bar(model_names, ll_vals, color=palette, alpha=0.85, edgecolor="white")
        ax.set_title("Log-Loss  (lower = better)", fontweight="bold")
        ax.set_ylabel("Log-Loss")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.tick_params(axis="x", rotation=15)

        # ── 3. Calibration curve ─────────────────────
        ax = axes[2]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Perfect")
        for name, color in zip(model_names, palette):
            P = self.results_[name]["predictions"]
            Y = self.results_[name]["actuals"]
            win_probs  = P[:, 0]
            win_actual = (Y == 0).astype(int)
            try:
                frac, mean_p = calibration_curve(win_actual, win_probs, n_bins=5)
                ax.plot(mean_p, frac, "s-", color=color, label=name, alpha=0.85, markersize=5)
            except Exception:
                pass
        ax.set_title("Calibration  — Win Probability", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual fraction")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        # ── 4. Per-class accuracy ────────────────────
        ax = axes[3]
        labels = ["Win", "Draw", "Loss"]
        x = np.arange(len(labels))
        w = 0.8 / len(model_names)
        for i, (name, color) in enumerate(zip(model_names, palette)):
            P = self.results_[name]["predictions"]
            Y = self.results_[name]["actuals"]
            accs = []
            for cls in range(3):
                mask = Y == cls
                if mask.sum() > 0:
                    accs.append((np.argmax(P[mask], axis=1) == cls).mean())
                else:
                    accs.append(0.0)
            ax.bar(x + i * w, accs, w, label=name, color=color, alpha=0.85)
        ax.set_title("Accuracy by Result Type", fontweight="bold")
        ax.set_xticks(x + w * (len(model_names) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved → {save_path}")
        plt.tight_layout()
        plt.show()

    # ─────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────

    def _compute_all_metrics(
        self, P: np.ndarray, Y: np.ndarray
    ) -> dict:
        """
        Args:
            P: (n, 3) array of [p_win, p_draw, p_loss]
            Y: (n,)  array of result labels 0/1/2
        """
        n = len(Y)
        Y_oh = np.zeros((n, 3))
        Y_oh[np.arange(n), Y] = 1.0

        acc = float(np.mean(np.argmax(P, axis=1) == Y))
        ll  = log_loss(Y, P)
        rps = self._rps(P, Y_oh)

        brier_w = brier_score_loss(Y_oh[:, 0], P[:, 0])
        brier_d = brier_score_loss(Y_oh[:, 1], P[:, 1])
        brier_l = brier_score_loss(Y_oh[:, 2], P[:, 2])

        return {
            "accuracy":  acc,
            "log_loss":  ll,
            "rps":       rps,
            "brier_win":  brier_w,
            "brier_draw": brier_d,
            "brier_loss": brier_l,
            "brier_avg":  (brier_w + brier_d + brier_l) / 3,
        }

    @staticmethod
    def _rps(P: np.ndarray, Y_oh: np.ndarray) -> float:
        """
        Ranked Probability Score for K-class ordinal outcomes.

        RPS = (1/(K-1)) × Σ_k (CDF_pred_k - CDF_actual_k)²

        For K=3 (Win/Draw/Loss), K-1 = 2 terms in the sum.
        """
        pred_cdf   = np.cumsum(P,    axis=1)
        actual_cdf = np.cumsum(Y_oh, axis=1)
        n_classes  = P.shape[1]
        rps_per    = np.sum((pred_cdf - actual_cdf) ** 2, axis=1) / (n_classes - 1)
        return float(np.mean(rps_per))
