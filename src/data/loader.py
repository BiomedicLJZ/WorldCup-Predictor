"""
src/data/loader.py

Downloads and preprocesses international football results from
martj42/international-football-results (GitHub), the most complete
open dataset of national team matches (1872 → today).
"""

import pandas as pd
import numpy as np
import requests
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Tournament importance weights: how much this result should move Elo ratings
# and how predictive historical performance in this competition is
TOURNAMENT_IMPORTANCE: dict[str, float] = {
    "FIFA World Cup": 1.00,
    "UEFA Euro": 0.85,
    "Copa América": 0.85,
    "AFC Asian Cup": 0.75,
    "Africa Cup of Nations": 0.75,
    "CONCACAF Gold Cup": 0.70,
    "FIFA Confederations Cup": 0.80,
    "UEFA Nations League": 0.65,
    "FIFA World Cup qualification": 0.60,
    "UEFA Euro qualification": 0.55,
    "Friendly": 0.25,
}


class DataLoader:
    """
    Handles downloading and initial cleaning of international football data.

    Primary source:
        github.com/martj42/international-football-results
        results.csv — all international matches since 1872
    """

    RAW_URL = (
        "https://raw.githubusercontent.com/martj42/"
        "international_results/master/results.csv"
    )

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────

    def download_data(self, force: bool = False) -> None:
        """Download raw results CSV from GitHub."""
        dest = self.data_dir / "results.csv"
        if dest.exists() and not force:
            logger.info("results.csv already exists. Pass force=True to re-download.")
            return

        logger.info("Downloading international results from GitHub…")
        r = requests.get(self.RAW_URL, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        logger.info(f"Saved → {dest}  ({dest.stat().st_size / 1024:.1f} KB)")

    def load_results(
        self,
        start_year: int = 1990,
        include_friendlies: bool = True,
    ) -> pd.DataFrame:
        """
        Load, clean, and enrich the results file.

        Returns a DataFrame with columns:
            date, home_team, away_team, home_score, away_score,
            tournament, neutral, importance_weight,
            result, result_numeric
        """
        path = self.data_dir / "results.csv"
        if not path.exists():
            self.download_data()

        df = pd.read_csv(path, parse_dates=["date"])

        # ── filters ──────────────────────────────────
        df = df[df["date"].dt.year >= start_year].copy()
        if not include_friendlies:
            df = df[df["tournament"] != "Friendly"]

        # ── cleaning ─────────────────────────────────
        df = df.dropna(subset=["home_score", "away_score"]).copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        # ── derived columns ───────────────────────────
        df["importance_weight"] = df["tournament"].map(self._importance)

        # Result from home-team perspective (0=Win, 1=Draw, 2=Loss)
        conditions = [
            df["home_score"] > df["away_score"],
            df["home_score"] == df["away_score"],
        ]
        df["result"] = np.select(conditions, ["W", "D"], default="L")
        df["result_numeric"] = np.select(conditions, [0, 1], default=2)

        df = df.sort_values("date").reset_index(drop=True)
        logger.info(
            f"Loaded {len(df):,} matches "
            f"({df['date'].min().year}–{df['date'].max().year})"
        )
        return df

    def load_world_cup_only(self, start_year: int = 1994) -> pd.DataFrame:
        """Convenience: only FIFA World Cup matches (for backtesting)."""
        df = self.load_results(start_year=start_year, include_friendlies=False)
        return df[df["tournament"] == "FIFA World Cup"].copy()

    def get_all_teams(self, df: pd.DataFrame) -> list[str]:
        teams = set(df["home_team"]) | set(df["away_team"])
        return sorted(teams)

    # ─────────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────────

    def _importance(self, tournament: str) -> float:
        t_lower = tournament.lower()
        for key, w in TOURNAMENT_IMPORTANCE.items():
            if key.lower() in t_lower:
                return w
        if "qualif" in t_lower:
            return 0.55
        if "friendly" in t_lower:
            return 0.25
        return 0.50
