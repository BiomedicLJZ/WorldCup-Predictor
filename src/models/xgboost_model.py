"""
src/models/xgboost_model.py

XGBoost multiclass classifier for W/D/L prediction.

Design choices:
- TimeSeriesSplit (never mix future into training fold)
- Optuna for automated hyperparameter search
- softprob objective → calibrated probabilities for W, D, L
- Feature importance tracked for interpretability
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from typing import Optional
import logging

from .base_model import BaseMatchPredictor, MatchPrediction
from ..features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostPredictor(BaseMatchPredictor):
    """
    Gradient-boosted trees model for match outcome prediction.

    Input:  tabular feature vector (Elo diff, form, H2H, etc.)
    Output: P(Win), P(Draw), P(Loss)

    Strengths:
    - Works well on small tabular datasets (WC has ~900 total matches)
    - Interpretable via feature importance
    - No distributional assumptions

    Weaknesses:
    - Cannot reason about unstructured information (injuries, news)
    - Does not explicitly model goal distributions
    """

    @property
    def name(self) -> str:
        return "XGBoost"

    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        n_optuna_trials: int = 50,
        n_cv_splits: int = 5,
    ):
        self.fe = feature_engineer
        self.n_optuna_trials = n_optuna_trials
        self.n_cv_splits = n_cv_splits

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_columns: list[str] = []
        self.feature_importance_: Optional[pd.DataFrame] = None
        self._df_train: Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """
        Train XGBoost on all matches in df.
        df must already have feature columns (output of FeatureEngineer).
        """
        self._df_train = df.copy()

        # Only use numeric feature columns that actually exist
        available = set(df.columns)
        self.feature_columns = [
            c for c in self.fe.get_feature_columns()
            if c in available
        ]

        X = df[self.feature_columns].fillna(0.0).values
        y = df["result_numeric"].values  # 0=Win, 1=Draw, 2=Loss

        logger.info(
            f"Training XGBoost on {len(X)} matches "
            f"with {len(self.feature_columns)} features"
        )
        logger.info(
            f"Class distribution — "
            f"Win: {np.sum(y==0)}, Draw: {np.sum(y==1)}, Loss: {np.sum(y==2)}"
        )

        best_params = self._optuna_search(X, y)

        self.model = xgb.XGBClassifier(
            **best_params,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        self.model.fit(X, y)

        self.feature_importance_ = (
            pd.DataFrame({
                "feature": self.feature_columns,
                "importance": self.model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        logger.info("XGBoost training complete.")
        logger.info(f"Top-5 features:\n{self.feature_importance_.head(5).to_string(index=False)}")

    # ─────────────────────────────────────────────────
    # PREDICTION
    # ─────────────────────────────────────────────────

    def predict(
        self,
        home_team: str,
        away_team: str,
        features: Optional[dict] = None,
        **kwargs,
    ) -> MatchPrediction:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if features is None:
            features = self.fe.build_single_match_features(home_team, away_team)

        x = np.array([[features.get(c, 0.0) for c in self.feature_columns]])
        probs = self.model.predict_proba(x)[0]  # [p_win, p_draw, p_loss]

        return MatchPrediction(
            p_home_win=float(probs[0]),
            p_draw=float(probs[1]),
            p_away_win=float(probs[2]),
            # Use scoring averages as proxy for expected goals
            lambda_home=features.get("avg_goals_scored_home", 1.3),
            lambda_away=features.get("avg_goals_scored_away", 1.1),
            metadata={"model": self.name},
        )

    # ─────────────────────────────────────────────────
    # OPTUNA HYPERPARAMETER SEARCH
    # ─────────────────────────────────────────────────

    def _optuna_search(self, X: np.ndarray, y: np.ndarray) -> dict:
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
                "max_depth":         trial.suggest_int("max_depth", 3, 6),
                "learning_rate":     trial.suggest_float("lr", 0.01, 0.15, log=True),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("col_btree", 0.6, 1.0),
                "min_child_weight":  trial.suggest_int("min_child_w", 1, 15),
                "gamma":             trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 3.0),
            }
            scores = []
            for tr_idx, va_idx in tscv.split(X):
                m = xgb.XGBClassifier(
                    **params,
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    random_state=42,
                    verbosity=0,
                )
                m.fit(X[tr_idx], y[tr_idx])
                probs = m.predict_proba(X[va_idx])
                scores.append(log_loss(y[va_idx], probs))
            return float(np.mean(scores))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=True)

        logger.info(f"Optuna best log-loss: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        return study.best_params
