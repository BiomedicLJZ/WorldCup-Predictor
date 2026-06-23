"""
src/models/poisson_model.py

Dixon-Coles (1997) Bivariate Poisson model for football score prediction.

Reference:
    Dixon, M. J., & Coles, S. G. (1997).
    Modelling association football scores and inefficiencies in the
    football betting market.
    Journal of the Royal Statistical Society: Series C, 46(2), 265–280.

Model:
    Each team has an ATTACK (α) and DEFENSE (δ) parameter.
    For a match between home H and away A:

        λ_H = exp(α_H  +  δ_A  +  γ)      ← expected goals home
        λ_A = exp(α_A  +  δ_H)             ← expected goals away

    γ is the home-venue advantage (in log-scale).

    The joint probability of score (x, y) is:
        P(X=x, Y=y) = τ(x,y,λ,μ,ρ) × Poisson(x;λ) × Poisson(y;μ)

    where τ is the Dixon-Coles low-score correction for correlation:
        τ(0,0) = 1 - λμρ
        τ(1,0) = 1 + μρ
        τ(0,1) = 1 + λρ
        τ(1,1) = 1 - ρ
        τ(x,y) = 1  otherwise

    Parameters are estimated via Maximum Likelihood (L-BFGS-B).

Strengths:
    - Full score distribution → W/D/L AND expected scorelines
    - Mathematically elegant, interpretable parameters
    - Standard in academic literature, easily publishable

Weaknesses:
    - Assumes goal-scoring rate is constant within a match
    - Ignores within-squad heterogeneity (red cards, injuries)
    - Needs sufficient historical data per team
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from typing import Optional
import logging
import warnings

from .base_model import BaseMatchPredictor, MatchPrediction

logger = logging.getLogger(__name__)


class DixonColesPredictor(BaseMatchPredictor):
    """
    Bivariate Poisson model with Dixon-Coles low-score correction.

    MLE is performed using L-BFGS-B on the full historical dataset
    with optional exponential time decay (recent matches count more).
    """

    @property
    def name(self) -> str:
        return "Dixon-Coles"

    def __init__(self, xi: float = 0.0018, max_goals: int = 8):
        """
        Args:
            xi:        Time-decay rate (ξ). Each extra day in the past
                       weights a match by exp(-ξ × days_ago).
                       0.0018 ≈ a match 1 year old weighs ~51% of a current one.
                       0 = no decay (all history weighted equally).
            max_goals: Max goals per team considered in the score matrix.
                       8 captures >99.9% of real-world outcomes.
        """
        self.xi = xi
        self.max_goals = max_goals

        # Learned parameters
        self.attack:    dict[str, float] = {}
        self.defense:   dict[str, float] = {}
        self.home_adv:  float = 0.0        # log-scale home advantage
        self.rho:       float = 0.0        # low-score correction

        self.teams_: list[str] = []
        self._df_train: Optional[pd.DataFrame] = None
        self._ref_date: Optional[pd.Timestamp] = None

    # ─────────────────────────────────────────────────
    # TRAINING — MLE
    # ─────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit Dixon-Coles parameters via MLE on historical match data.
        Identifiability: first team's attack is fixed at 0 (reference level).
        """
        self._df_train = df.copy()
        self._ref_date = df["date"].max()

        self.teams_ = sorted(set(df["home_team"]) | set(df["away_team"]))
        n = len(self.teams_)
        idx = {t: i for i, t in enumerate(self.teams_)}

        logger.info(f"Fitting Dixon-Coles: {n} teams, {len(df)} matches…")

        # Time-decay weights
        days_ago = (self._ref_date - df["date"]).dt.days.values
        w = np.exp(-self.xi * days_ago)

        # Precompute integer arrays (faster in the loss loop)
        hi = df["home_team"].map(idx).values
        ai = df["away_team"].map(idx).values
        hg = df["home_score"].values.astype(int)
        ag = df["away_score"].values.astype(int)
        neutral = df["neutral"].values.astype(float)

        # Precompute log factorial for exact log-likelihood values
        from scipy.special import gammaln
        log_fact_hg = gammaln(hg + 1)
        log_fact_ag = gammaln(ag + 1)

        # Parameter layout:
        #   [0 .. n-1]        attack   (α)  — α[0] fixed at 0
        #   [n .. 2n-1]       defense  (δ)
        #   [2n]              home adv (γ)
        #   [2n+1]            rho      (ρ)
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.1]])

        def neg_ll(params: np.ndarray) -> float:
            alpha  = params[:n]
            delta  = params[n:2*n]
            gamma  = params[2*n]
            rho    = params[2*n + 1]

            log_lam_h = alpha[hi] + delta[ai] + gamma * (1 - neutral)
            log_lam_a = alpha[ai] + delta[hi]

            lam_h = np.exp(log_lam_h)
            lam_a = np.exp(log_lam_a)

            tau   = self._tau_vec(hg, ag, lam_h, lam_a, rho)
            
            ll_h = hg * log_lam_h - lam_h - log_fact_hg
            ll_a = ag * log_lam_a - lam_a - log_fact_ag

            ll    = np.log(np.clip(tau, 1e-12, None)) + ll_h + ll_a
            return -float(np.dot(w, ll))

        # Identifiability: α[0] is pinned to 0 via the (0, 0) bound below.
        # (No in-place mutation of `params` — that would corrupt L-BFGS-B's
        #  finite-difference gradient estimate.)
        bounds = (
            [(0, 0)]               +  # α[0] = 0 fixed
            [(None, None)] * (n-1) +  # α[1..n-1]
            [(None, None)] * n     +  # δ
            [(None, None)]         +  # γ
            [(-0.99, 0.99)]           # ρ
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                neg_ll,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-10},
            )

        if not res.success:
            logger.warning(f"MLE optimizer: {res.message}")
        else:
            logger.info(f"MLE converged in {res.nit} iterations.")

        params = res.x
        self.attack   = {t: params[i]       for t, i in idx.items()}
        self.defense  = {t: params[n + i]   for t, i in idx.items()}
        self.home_adv = params[2*n]
        self.rho      = params[2*n + 1]

        logger.info(f"Home advantage multiplier: {np.exp(self.home_adv):.3f}x")
        logger.info(f"Rho (low-score correction): {self.rho:.4f}")
        top5 = sorted(self.attack.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Top-5 attacking teams: {top5}")

    # ─────────────────────────────────────────────────
    # PREDICTION
    # ─────────────────────────────────────────────────

    def predict(
        self,
        home_team: str,
        away_team: str,
        features: Optional[dict] = None,
        is_neutral: bool = True,
        **kwargs,
    ) -> MatchPrediction:
        """
        Predict via the full joint score-probability matrix.
        Unknown teams fall back to the average parameter across all teams.
        """
        if not self.attack:
            raise RuntimeError("Model not fitted. Call fit() first.")

        avg_atk = float(np.mean(list(self.attack.values())))
        avg_def = float(np.mean(list(self.defense.values())))

        a_h = self.attack.get(home_team, avg_atk)
        d_h = self.defense.get(home_team, avg_def)
        a_a = self.attack.get(away_team, avg_atk)
        d_a = self.defense.get(away_team, avg_def)

        home_adv = 0.0 if is_neutral else self.home_adv

        lam_h = np.exp(a_h + d_a + home_adv)
        lam_a = np.exp(a_a + d_h)

        mat = self._score_matrix(lam_h, lam_a)

        p_win  = float(np.sum(np.tril(mat, -1)))   # home scores more
        p_draw = float(np.sum(np.diag(mat)))
        p_loss = float(np.sum(np.triu(mat, 1)))

        total = p_win + p_draw + p_loss
        return MatchPrediction(
            p_home_win=p_win  / total,
            p_draw=    p_draw / total,
            p_away_win=p_loss / total,
            lambda_home=lam_h,
            lambda_away=lam_a,
            metadata={
                "model": self.name,
                "α_home": a_h, "δ_home": d_h,
                "α_away": a_a, "δ_away": d_a,
            },
        )

    def get_team_parameters(self) -> pd.DataFrame:
        """Return a DataFrame of team attack/defense strengths."""
        rows = []
        for team in self.teams_:
            a = self.attack.get(team, 0.0)
            d = self.defense.get(team, 0.0)
            rows.append({
                "team":          team,
                "attack_log":    a,
                "defense_log":   d,
                "attack_mult":   np.exp(a),   # multiplier vs average team
                "defense_mult":  np.exp(d),   # lower defense → stronger defense
                "net_strength":  a - d,       # combined offensive minus defensive weakness
            })
        return (
            pd.DataFrame(rows)
            .sort_values("net_strength", ascending=False)
            .reset_index(drop=True)
        )

    # ─────────────────────────────────────────────────
    # MATH HELPERS
    # ─────────────────────────────────────────────────

    def _score_matrix(self, lam_h: float, lam_a: float) -> np.ndarray:
        g = self.max_goals + 1
        goals = np.arange(g)
        # Vectorized outer product — ~100x faster than the Python double loop
        mat = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
        # Dixon-Coles low-score correction on the four special cells only
        rho = self.rho
        mat[0, 0] = max(1e-12, mat[0, 0] * (1 - lam_h * lam_a * rho))
        mat[0, 1] = max(1e-12, mat[0, 1] * (1 + lam_h * rho))
        mat[1, 0] = max(1e-12, mat[1, 0] * (1 + lam_a * rho))
        mat[1, 1] = max(1e-12, mat[1, 1] * (1 - rho))
        return mat

    def simulate_score_distribution(
        self,
        home_team: str,
        away_team: str,
        features=None,
        is_neutral: bool = True,
        n_sim: int = 50_000,  # kept for interface compat; exact method ignores it
    ) -> dict:
        """Return exact score probabilities from the DC score matrix (no sampling)."""
        if not self.attack:
            return super().simulate_score_distribution(home_team, away_team, features, n_sim)
        pred = self.predict(home_team, away_team, is_neutral=is_neutral)
        mat  = self._score_matrix(pred.lambda_home, pred.lambda_away)
        g    = self.max_goals + 1
        scores = sorted(
            [(f"{i}-{j}", float(mat[i, j])) for i in range(g) for j in range(g)],
            key=lambda x: x[1], reverse=True,
        )
        return {
            "prediction":  pred,
            "top_scores":  scores[:10],
            "avg_xg_home": float(pred.lambda_home),
            "avg_xg_away": float(pred.lambda_away),
        }

    @staticmethod
    def _tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
        if   x == 0 and y == 0: return 1 - lh * la * rho
        elif x == 0 and y == 1: return 1 + lh * rho
        elif x == 1 and y == 0: return 1 + la * rho
        elif x == 1 and y == 1: return 1 - rho
        return 1.0

    @staticmethod
    def _tau_vec(x: np.ndarray, y: np.ndarray,
                 lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
        tau = np.ones(len(x))
        tau[(x == 0) & (y == 0)] = 1 - lh[(x==0)&(y==0)] * la[(x==0)&(y==0)] * rho
        tau[(x == 0) & (y == 1)] = 1 + lh[(x==0)&(y==1)] * rho
        tau[(x == 1) & (y == 0)] = 1 + la[(x==1)&(y==0)] * rho
        tau[(x == 1) & (y == 1)] = 1 - rho
        return tau
