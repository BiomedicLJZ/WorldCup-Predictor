"""
src/features/elo.py

Custom Elo rating system for national football teams.

Inspired by World Football Elo Ratings (eloratings.net) methodology,
with configurable K-factor, home advantage, and goal-margin bonus.

Key insight: Elo difference is consistently the single most predictive
feature in football match prediction models. All other features are
refinements on top of this foundation.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EloSystem:
    """
    Elo rating system with:
    - Variable K-factor scaled by tournament importance
    - Optional home-venue advantage correction
    - Goal-margin multiplier (diminishing returns)
    - Full historical rating snapshots (for backtest data integrity)
    """

    DEFAULT_RATING = 1500.0

    # Goal-margin multiplier table (capped at 4 goals)
    MARGIN_MULT = {1: 1.00, 2: 1.50, 3: 1.75, 4: 1.875}

    def __init__(
        self,
        k_base: float = 40,
        default_rating: float = 1500.0,
        home_advantage: float = 100.0,
    ):
        """
        Args:
            k_base: K-factor for a FIFA World Cup match.
                    Friendly = k_base × 0.25, qualifier = k_base × 0.60
            default_rating: Starting Elo for any team (1500 = average)
            home_advantage: Elo-point bonus added to home team's rating
                            before computing expected outcome. 100 ≈ 64% win
                            probability for equally-rated teams at home.
        """
        self.k_base = k_base
        self.default_rating = default_rating
        self.home_advantage = home_advantage

        self.ratings: dict[str, float] = defaultdict(lambda: self.default_rating)
        # history[team] = [(pd.Timestamp, rating), ...]  chronologically sorted
        self.history: dict[str, list[tuple]] = defaultdict(list)

    # ─────────────────────────────────────────────────
    # CORE COMPUTATION
    # ─────────────────────────────────────────────────

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all matches in chronological order, updating ratings.

        Adds four columns to df:
            elo_home_pre  – home team Elo BEFORE this match
            elo_away_pre  – away team Elo BEFORE this match
            elo_home_post – home team Elo AFTER this match
            elo_away_post – away team Elo AFTER this match
            elo_diff      – elo_home_pre - elo_away_pre
        """
        df = df.sort_values("date").reset_index(drop=True)

        pre_home, pre_away, post_home, post_away = [], [], [], []

        for _, row in df.iterrows():
            h, a = row["home_team"], row["away_team"]

            r_h = self.ratings[h]
            r_a = self.ratings[a]
            pre_home.append(r_h)
            pre_away.append(r_a)

            # Apply home-venue bonus (skip if neutral ground)
            r_h_adj = r_h + (0.0 if row.get("neutral", False) else self.home_advantage)

            exp_h = self._expected(r_h_adj, r_a)
            act_h, act_a = self._outcome(row["home_score"], row["away_score"])

            k = self.k_base * row.get("importance_weight", 0.50)
            margin_mult = self._margin_mult(abs(row["home_score"] - row["away_score"]))

            delta_h = k * margin_mult * (act_h - exp_h)
            delta_a = -delta_h  # Zero-sum

            new_h = r_h + delta_h
            new_a = r_a + delta_a
            self.ratings[h] = new_h
            self.ratings[a] = new_a

            self.history[h].append((row["date"], new_h))
            self.history[a].append((row["date"], new_a))

            post_home.append(new_h)
            post_away.append(new_a)

        df["elo_home_pre"] = pre_home
        df["elo_away_pre"] = pre_away
        df["elo_home_post"] = post_home
        df["elo_away_post"] = post_away
        df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]

        logger.info(f"Elo computed for {len(self.ratings)} teams.")
        return df

    # ─────────────────────────────────────────────────
    # QUERYING
    # ─────────────────────────────────────────────────

    def get_rating(self, team: str, date: Optional[pd.Timestamp] = None) -> float:
        """
        Return team Elo at a given date.
        If date is None, returns the current (latest) rating.
        Uses linear scan from the end for simplicity (dataset is small enough).
        """
        if date is None:
            return self.ratings.get(team, self.default_rating)

        history = self.history.get(team, [])
        for ts, rating in reversed(history):
            if ts <= date:
                return rating
        return self.default_rating

    def get_top_teams(self, n: int = 20, date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Return top N teams by Elo rating as a DataFrame."""
        if date is None:
            rating_dict = dict(self.ratings)
        else:
            rating_dict = {t: self.get_rating(t, date) for t in self.ratings}

        df = pd.DataFrame(rating_dict.items(), columns=["team", "elo"])
        return df.nlargest(n, "elo").reset_index(drop=True)

    def get_expected_win_prob(self, team_a: str, team_b: str) -> float:
        """P(team_a wins) based solely on current Elo ratings."""
        return self._expected(self.ratings[team_a], self.ratings[team_b])

    # ─────────────────────────────────────────────────
    # MATH HELPERS
    # ─────────────────────────────────────────────────

    def _expected(self, r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))

    @staticmethod
    def _outcome(gh: int, ga: int) -> tuple[float, float]:
        if gh > ga:
            return 1.0, 0.0
        if gh < ga:
            return 0.0, 1.0
        return 0.5, 0.5

    def _margin_mult(self, margin: int) -> float:
        return self.MARGIN_MULT.get(min(margin, 4), 1.875)
