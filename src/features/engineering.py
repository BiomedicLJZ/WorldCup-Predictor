"""
src/features/engineering.py

Builds the full tabular feature matrix used by all models.

⚠️ DATA INTEGRITY RULE: Every feature for a match at date T
   must use ONLY matches with date < T. No data leakage.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging
from .elo import EloSystem

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# CONFEDERATION MAPPINGS
# ─────────────────────────────────────────────────────────

CONFEDERATION_MAP: dict[str, str] = {
    # UEFA
    **{t: "UEFA" for t in [
        "Germany", "France", "Spain", "Portugal", "England", "Netherlands",
        "Belgium", "Croatia", "Italy", "Switzerland", "Denmark", "Sweden",
        "Poland", "Hungary", "Czech Republic", "Austria", "Serbia", "Wales",
        "Scotland", "Ukraine", "Turkey", "Slovakia", "Slovenia", "Albania",
        "Romania", "Greece", "Norway", "Finland", "Russia", "Iceland",
        "Bosnia and Herzegovina", "North Macedonia", "Montenegro", "Bulgaria",
        "Cyprus", "Latvia", "Lithuania", "Estonia", "Andorra", "Malta",
        "Liechtenstein", "San Marino", "Kosovo", "Georgia", "Armenia",
        "Azerbaijan", "Kazakhstan", "Moldova", "Luxembourg", "Belarus",
    ]},
    # CONMEBOL
    **{t: "CONMEBOL" for t in [
        "Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Ecuador",
        "Peru", "Venezuela", "Paraguay", "Bolivia",
    ]},
    # CONCACAF
    **{t: "CONCACAF" for t in [
        "Mexico", "United States", "Canada", "Costa Rica", "Honduras",
        "Jamaica", "Panama", "El Salvador", "Haiti", "Trinidad and Tobago",
        "Cuba", "Guatemala", "Nicaragua", "Dominican Republic", "Belize",
        "Suriname", "Guyana", "Barbados", "Antigua and Barbuda",
        "Curacao",  # WC 2026 qualifier
    ]},
    # AFC
    **{t: "AFC" for t in [
        "Japan", "South Korea", "Iran", "Saudi Arabia", "Australia",
        "Qatar", "Iraq", "United Arab Emirates", "Jordan", "China",
        "Uzbekistan", "Vietnam", "Thailand", "Oman", "Bahrain",
        "Kuwait", "Syria", "Palestine", "Lebanon", "India",
        "Indonesia", "Malaysia", "Philippines",
    ]},
    # CAF
    **{t: "CAF" for t in [
        "Morocco", "Senegal", "Nigeria", "Ghana", "Cameroon", "Egypt",
        "Tunisia", "Ivory Coast", "Mali", "South Africa", "Algeria",
        "DR Congo", "Kenya", "Zimbabwe", "Mozambique", "Tanzania",
        "Uganda", "Rwanda", "Zambia", "Angola", "Ethiopia", "Burkina Faso",
        "Guinea", "Cape Verde", "Gabon", "Libya",
    ]},
    # OFC
    **{t: "OFC" for t in [
        "New Zealand", "Fiji", "Papua New Guinea", "Solomon Islands",
        "Vanuatu", "Tahiti",
    ]},
}

CONFEDERATION_STRENGTH: dict[str, float] = {
    "UEFA": 1.000,
    "CONMEBOL": 0.950,
    "AFC": 0.700,
    "CAF": 0.685,
    "CONCACAF": 0.660,
    "OFC": 0.500,
    "Unknown": 0.600,
}


# ─────────────────────────────────────────────────────────
# FEATURE ENGINEER
# ─────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Transforms a raw match DataFrame (enriched with Elo) into a
    feature matrix ready for ML training.

    All features respect the temporal boundary — each row only uses
    information available strictly before that match's date.
    """

    # These are the numeric columns that go into the ML models
    NUMERIC_FEATURES = [
        "elo_diff", "elo_home_pre", "elo_away_pre",
        "form_home", "form_away", "form_diff",
        "avg_goals_scored_home", "avg_goals_conceded_home",
        "avg_goals_scored_away", "avg_goals_conceded_away",
        "h2h_win_rate_home", "h2h_total_games",
        "is_neutral",
        "conf_strength_home", "conf_strength_away", "conf_strength_diff",
        # Momentum features (v2): same math drives both build_features and _match_features
        "win_streak_home", "win_streak_away",
        "goal_trend_home", "goal_trend_away",
        "clean_sheet_rate_home", "clean_sheet_rate_away",
    ]

    def __init__(self, elo_system: EloSystem, form_window: int = 10):
        self.elo = elo_system
        self.form_window = form_window
        self._all_matches: Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the full feature matrix on a match DataFrame that
        already has elo_home_pre / elo_away_pre / elo_diff columns
        (produced by EloSystem.compute_all).
        """
        self._all_matches = df.copy()
        logger.info(f"Building features for {len(df)} matches (optimized)…")

        from collections import defaultdict
        
        # State tracking
        team_hist = defaultdict(list)  # team -> list of (gf, ga, pts)
        h2h_hist = defaultdict(list)   # tuple(sorted(t1, t2)) -> list of winner
        
        rows = []
        df_sorted = df.sort_values("date")
        
        for idx, row in df_sorted.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            
            # 1. Compute Features from State
            def get_team_stats(team):
                hist = team_hist[team][-self.form_window:]
                if not hist:
                    return 1.0, 1.3, 1.1
                pts = [x[2] for x in hist]
                gf = [x[0] for x in hist]
                ga = [x[1] for x in hist]
                return float(np.mean(pts)), float(np.mean(gf)), float(np.mean(ga))

            form_h, gs_h, gc_h = get_team_stats(home)
            form_a, gs_a, gc_a = get_team_stats(away)
            
            pair = tuple(sorted([home, away]))
            pair_hist = h2h_hist[pair][-10:]
            if not pair_hist:
                h2h_win_rate_home = 0.33
                h2h_total = 0
            else:
                h2h_total = len(pair_hist)
                h2h_win_rate_home = pair_hist.count(home) / h2h_total

            conf_h = CONFEDERATION_MAP.get(home, "Unknown")
            conf_a = CONFEDERATION_MAP.get(away, "Unknown")
            str_h = CONFEDERATION_STRENGTH.get(conf_h, CONFEDERATION_STRENGTH["Unknown"])
            str_a = CONFEDERATION_STRENGTH.get(conf_a, CONFEDERATION_STRENGTH["Unknown"])

            feat = {
                "elo_diff": row.get("elo_diff", 0.0),
                "elo_home_pre": row.get("elo_home_pre", 1500.0),
                "elo_away_pre": row.get("elo_away_pre", 1500.0),
                "form_home": form_h,
                "form_away": form_a,
                "form_diff": form_h - form_a,
                "avg_goals_scored_home": gs_h,
                "avg_goals_conceded_home": gc_h,
                "avg_goals_scored_away": gs_a,
                "avg_goals_conceded_away": gc_a,
                "h2h_win_rate_home": h2h_win_rate_home,
                "h2h_total_games": h2h_total,
                "is_neutral": int(row.get("neutral", True)),
                "conf_home": conf_h,
                "conf_away": conf_a,
                "conf_strength_home": str_h,
                "conf_strength_away": str_a,
                "conf_strength_diff": str_h - str_a,
                # Momentum features — driven by the same static helpers used in _match_features
                "win_streak_home":      self._streak_from_list(team_hist[home][-self.form_window:]),
                "win_streak_away":      self._streak_from_list(team_hist[away][-self.form_window:]),
                "goal_trend_home":      self._trend_from_list(team_hist[home][-self.form_window:]),
                "goal_trend_away":      self._trend_from_list(team_hist[away][-self.form_window:]),
                "clean_sheet_rate_home": self._cs_rate_from_list(team_hist[home][-self.form_window:]),
                "clean_sheet_rate_away": self._cs_rate_from_list(team_hist[away][-self.form_window:]),
            }
            rows.append(feat)

            # 2. Update State
            hs = row.get("home_score")
            as_ = row.get("away_score")
            if pd.notna(hs) and pd.notna(as_):
                hs, as_ = int(hs), int(as_)
                if hs > as_:
                    pts_h, pts_a, winner = 3, 0, home
                elif hs == as_:
                    pts_h, pts_a, winner = 1, 1, "Draw"
                else:
                    pts_h, pts_a, winner = 0, 3, away

                team_hist[home].append((hs, as_, pts_h))
                team_hist[away].append((as_, hs, pts_a))
                h2h_hist[pair].append(winner)

        feat_df = pd.DataFrame(rows, index=df_sorted.index)
        
        # Merge back
        result = df.copy()
        for col in feat_df.columns:
            result[col] = feat_df[col]

        logger.info(f"Feature matrix ready: {feat_df.shape[1]} feature columns.")
        return result

    def build_single_match_features(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = True,
        reference_date: Optional[pd.Timestamp] = None,
    ) -> dict:
        """
        Compute features for a FUTURE / hypothetical match.
        Uses all available historical data (no cutoff needed).
        """
        if self._all_matches is None:
            raise RuntimeError("Call build_features() on training data first.")

        hist = self._all_matches
        date = reference_date or pd.Timestamp.now()

        elo_h = self.elo.get_rating(home_team, date)
        elo_a = self.elo.get_rating(away_team, date)

        fake_row = pd.Series({
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            "neutral": is_neutral,
            "elo_home_pre": elo_h,
            "elo_away_pre": elo_a,
            "elo_diff": elo_h - elo_a,
        })
        return self._match_features(fake_row, hist)

    def get_feature_columns(self) -> list[str]:
        return self.NUMERIC_FEATURES.copy()

    # ─────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────

    def _match_features(self, row: pd.Series, df: pd.DataFrame) -> dict:
        home, away = row["home_team"], row["away_team"]
        date = row["date"]
        historical = df[df["date"] < date]

        conf_h = CONFEDERATION_MAP.get(home, "Unknown")
        conf_a = CONFEDERATION_MAP.get(away, "Unknown")
        str_h = CONFEDERATION_STRENGTH.get(conf_h, CONFEDERATION_STRENGTH["Unknown"])
        str_a = CONFEDERATION_STRENGTH.get(conf_a, CONFEDERATION_STRENGTH["Unknown"])

        form_h = self._form(historical, home)
        form_a = self._form(historical, away)

        return {
            # Elo
            "elo_diff": row.get("elo_diff", 0.0),
            "elo_home_pre": row.get("elo_home_pre", 1500.0),
            "elo_away_pre": row.get("elo_away_pre", 1500.0),

            # Form (average points/game in last N matches)
            "form_home": form_h,
            "form_away": form_a,
            "form_diff": form_h - form_a,

            # Scoring averages
            "avg_goals_scored_home": self._avg_scored(historical, home),
            "avg_goals_conceded_home": self._avg_conceded(historical, home),
            "avg_goals_scored_away": self._avg_scored(historical, away),
            "avg_goals_conceded_away": self._avg_conceded(historical, away),

            # Head-to-head
            **self._h2h(historical, home, away),

            # Context
            "is_neutral": int(row.get("neutral", True)),

            # Confederation strength
            "conf_home": conf_h,
            "conf_away": conf_a,
            "conf_strength_home": str_h,
            "conf_strength_away": str_a,
            "conf_strength_diff": str_h - str_a,
            # Momentum features — same static helpers as build_features path
            "win_streak_home":       self._win_streak(historical, home),
            "win_streak_away":       self._win_streak(historical, away),
            "goal_trend_home":       self._goal_trend(historical, home),
            "goal_trend_away":       self._goal_trend(historical, away),
            "clean_sheet_rate_home": self._clean_sheet_rate(historical, home),
            "clean_sheet_rate_away": self._clean_sheet_rate(historical, away),
        }

    def _team_matches(self, df: pd.DataFrame, team: str, n: Optional[int] = None) -> pd.DataFrame:
        mask = (df["home_team"] == team) | (df["away_team"] == team)
        matches = df[mask]
        return matches.tail(n) if n else matches

    def _form(self, df: pd.DataFrame, team: str) -> float:
        matches = self._team_matches(df, team, self.form_window)
        if matches.empty:
            return 1.0  # neutral (= expected PPG for average team)
        pts = []
        for _, m in matches.iterrows():
            gf = m["home_score"] if m["home_team"] == team else m["away_score"]
            ga = m["away_score"] if m["home_team"] == team else m["home_score"]
            pts.append(3 if gf > ga else (1 if gf == ga else 0))
        return float(np.mean(pts))

    def _avg_scored(self, df: pd.DataFrame, team: str) -> float:
        matches = self._team_matches(df, team, self.form_window)
        if matches.empty:
            return 1.3
        goals = [m["home_score"] if m["home_team"] == team else m["away_score"]
                 for _, m in matches.iterrows()]
        return float(np.mean(goals))

    def _avg_conceded(self, df: pd.DataFrame, team: str) -> float:
        matches = self._team_matches(df, team, self.form_window)
        if matches.empty:
            return 1.1
        goals = [m["away_score"] if m["home_team"] == team else m["home_score"]
                 for _, m in matches.iterrows()]
        return float(np.mean(goals))

    def _h2h(self, df: pd.DataFrame, home: str, away: str) -> dict:
        mask = (
            ((df["home_team"] == home) & (df["away_team"] == away)) |
            ((df["home_team"] == away) & (df["away_team"] == home))
        )
        h2h = df[mask].tail(10)
        if h2h.empty:
            return {"h2h_win_rate_home": 0.33, "h2h_total_games": 0}

        wins = 0
        for _, m in h2h.iterrows():
            if m["home_team"] == home and m["home_score"] > m["away_score"]:
                wins += 1
            elif m["away_team"] == home and m["away_score"] > m["home_score"]:
                wins += 1
        return {"h2h_win_rate_home": wins / len(h2h), "h2h_total_games": len(h2h)}

    # ── Momentum helpers — static so both paths use identical math ────────

    @staticmethod
    def _streak_from_list(hist: list) -> int:
        """Consecutive wins at the tail of a (gf, ga, pts) list."""
        s = 0
        for _, _, pts in reversed(hist):
            if pts == 3:
                s += 1
            else:
                break
        return s

    @staticmethod
    def _trend_from_list(hist: list) -> float:
        """Linear slope of goals scored over the list (positive = improving attack)."""
        if len(hist) < 3:
            return 0.0
        goals = np.array([x[0] for x in hist], dtype=float)
        n = len(goals)
        x = np.arange(n, dtype=float) - (n - 1) / 2.0
        denom = float(np.dot(x, x))
        return float(np.dot(x, goals) / denom) if denom > 0 else 0.0

    @staticmethod
    def _cs_rate_from_list(hist: list) -> float:
        """Fraction of matches where the team conceded 0 goals."""
        if not hist:
            return 0.3
        return sum(1 for _, ga, _ in hist if ga == 0) / len(hist)

    # ── DataFrame-based wrappers (used by _match_features path) ───────────

    def _win_streak(self, df: pd.DataFrame, team: str) -> int:
        matches = self._team_matches(df, team, self.form_window)
        hist = [
            (
                int(m["home_score"] if m["home_team"] == team else m["away_score"]),
                int(m["away_score"] if m["home_team"] == team else m["home_score"]),
                3 if (m["home_score"] if m["home_team"] == team else m["away_score"]) >
                     (m["away_score"] if m["home_team"] == team else m["home_score"])
                else (1 if m["home_score"] == m["away_score"] else 0),
            )
            for _, m in matches.iterrows()
        ]
        return self._streak_from_list(hist)

    def _goal_trend(self, df: pd.DataFrame, team: str) -> float:
        matches = self._team_matches(df, team, self.form_window)
        hist = [
            (
                int(m["home_score"] if m["home_team"] == team else m["away_score"]),
                int(m["away_score"] if m["home_team"] == team else m["home_score"]),
                0,
            )
            for _, m in matches.iterrows()
        ]
        return self._trend_from_list(hist)

    def _clean_sheet_rate(self, df: pd.DataFrame, team: str) -> float:
        matches = self._team_matches(df, team, self.form_window)
        hist = [
            (
                0,
                int(m["away_score"] if m["home_team"] == team else m["home_score"]),
                0,
            )
            for _, m in matches.iterrows()
        ]
        return self._cs_rate_from_list(hist)
