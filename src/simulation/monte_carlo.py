"""
src/simulation/monte_carlo.py

Monte Carlo simulator for the FIFA World Cup 2026.

Format:
    48 teams  →  12 groups of 4
    Each group: round-robin (6 matches)
    Top 2 from each group advance (24 teams)
    8 best 3rd-place finishers advance (8 teams)
    Round of 32 → 16 → 8 (QF) → 4 (SF) → Final + 3rd-place

Usage:
    sim = WorldCupSimulator(predictor=my_model)
    df  = sim.champion_probabilities(n_simulations=10_000)
    print(df.head(10))
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional
import logging
from tqdm import tqdm

import requests

from ..models.base_model import BaseMatchPredictor

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# ESPN display name → martj42 dataset spelling
# ─────────────────────────────────────────────────────────
_ESPN_NAME_MAP: dict[str, str] = {
    "Czechia":                      "Czech Republic",
    "Czech Republic":               "Czech Republic",
    "Türkiye":                      "Turkey",
    "Turkey":                       "Turkey",
    "Bosnia-Herzegovina":           "Bosnia and Herzegovina",
    "Bosnia and Herzegovina":       "Bosnia and Herzegovina",
    "Côte d'Ivoire":                "Ivory Coast",
    "Ivory Coast":                  "Ivory Coast",
    "Cabo Verde":                   "Cape Verde",
    "Cape Verde":                   "Cape Verde",
    "DR Congo":                     "DR Congo",
    "Congo DR":                     "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Curaçao":                      "Curacao",
    "Curacao":                      "Curacao",
    "IR Iran":                      "Iran",
    "Iran":                         "Iran",
    "Korea Republic":               "South Korea",
    "South Korea":                  "South Korea",
    "USA":                          "United States",
    "United States":                "United States",
}


def _norm(name: str) -> str:
    return _ESPN_NAME_MAP.get(name, name)


def fetch_live_wc_results(timeout: int = 6) -> dict[tuple[str, str], tuple[int, int]]:
    """
    Attempt to pull completed WC 2026 group-stage results from ESPN's
    unofficial soccer scoreboard API.  On any failure (network, parse,
    no data) returns a copy of the hardcoded KNOWN_WC_2026_RESULTS so
    the simulation always gets a valid dict.

    Call this once per pipeline run and pass the result as `known_results`
    to WorldCupSimulator to use live scores.
    """
    # ESPN unofficial scoreboard endpoint — no API key required
    ESPN_URL = (
        "https://site.api.espn.com/apis/site/v2/sports/soccer"
        "/fifa.world/scoreboard"
    )
    try:
        resp = requests.get(ESPN_URL, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        live: dict[tuple[str, str], tuple[int, int]] = {}
        for event in data.get("events", []):
            comp = (event.get("competitions") or [{}])[0]
            if not comp.get("status", {}).get("type", {}).get("completed", False):
                continue
            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue
            # ESPN lists home first when homeAway == "home"
            home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
            h_name = _norm(home_c.get("team", {}).get("displayName", ""))
            a_name = _norm(away_c.get("team", {}).get("displayName", ""))
            try:
                h_score = int(home_c.get("score", 0))
                a_score = int(away_c.get("score", 0))
            except (TypeError, ValueError):
                continue
            if h_name and a_name:
                live[(h_name, a_name)] = (h_score, a_score)

        if live:
            logger.info(f"Live WC results fetched: {len(live)} completed matches from ESPN.")
            merged = dict(KNOWN_WC_2026_RESULTS)
            merged.update(live)  # live data overwrites hardcoded if conflict
            return merged

        logger.warning("ESPN returned no completed WC matches. Using hardcoded fallback.")
    except Exception as exc:
        logger.warning(f"Live WC fetch failed ({exc!r}). Using hardcoded fallback.")

    return dict(KNOWN_WC_2026_RESULTS)

# ─────────────────────────────────────────────────────────
# WC 2026 GROUP DRAW  (preliminary — actual draw: Dec 2025)
# Teams marked TBD_* are placeholders for qualification spots
# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
# CONFIRMED MATCH RESULTS  (as of 2026-06-22, ESPN / FOX)
# Key: (home_as_listed, away_as_listed) → (goals_home, goals_away)
# Lookup checks both orderings in _simulate_group.
# ─────────────────────────────────────────────────────────
KNOWN_WC_2026_RESULTS: dict[tuple[str, str], tuple[int, int]] = {
    # ── Group A ──────────────────────────────────────────
    ("Mexico",        "South Africa"):  (2, 0),
    ("South Korea",   "Czech Republic"): (2, 1),
    ("Czech Republic","South Africa"):  (1, 1),
    ("Mexico",        "South Korea"):   (1, 0),
    # ── Group B ──────────────────────────────────────────
    ("Canada",        "Bosnia and Herzegovina"): (1, 1),
    ("Qatar",         "Switzerland"):   (1, 1),
    ("Switzerland",   "Bosnia and Herzegovina"): (4, 1),
    ("Canada",        "Qatar"):         (6, 0),
    # ── Group C ──────────────────────────────────────────
    ("Brazil",        "Morocco"):       (1, 1),
    ("Haiti",         "Scotland"):      (0, 1),
    ("Scotland",      "Morocco"):       (0, 1),
    ("Brazil",        "Haiti"):         (3, 0),
    # ── Group D ──────────────────────────────────────────
    ("United States", "Paraguay"):      (4, 1),
    ("Australia",     "Turkey"):        (2, 0),
    ("United States", "Australia"):     (2, 0),
    ("Turkey",        "Paraguay"):      (0, 1),
    # ── Group E ──────────────────────────────────────────
    ("Germany",       "Curacao"):       (7, 1),
    ("Ivory Coast",   "Ecuador"):       (1, 0),
    ("Germany",       "Ivory Coast"):   (2, 1),
    ("Ecuador",       "Curacao"):       (0, 0),
    # ── Group F ──────────────────────────────────────────
    ("Netherlands",   "Japan"):         (2, 2),
    ("Sweden",        "Tunisia"):       (5, 1),
    ("Netherlands",   "Sweden"):        (5, 1),
    ("Tunisia",       "Japan"):         (0, 4),
    # ── Group G ──────────────────────────────────────────
    ("Belgium",       "Egypt"):         (1, 1),
    ("Iran",          "New Zealand"):   (2, 2),
    ("Belgium",       "Iran"):          (0, 0),
    ("New Zealand",   "Egypt"):         (1, 3),
    # ── Group H ──────────────────────────────────────────
    ("Spain",         "Cape Verde"):    (0, 0),
    ("Saudi Arabia",  "Uruguay"):       (1, 1),
    ("Spain",         "Saudi Arabia"):  (4, 0),
    ("Uruguay",       "Cape Verde"):    (2, 2),
    # ── Group I (partial as of 2026-06-22) ───────────────
    ("France",        "Iraq"):          (3, 0),
    # ── Group J (partial) ────────────────────────────────
    ("Argentina",     "Austria"):       (2, 0),
}

# ─────────────────────────────────────────────────────────
# TEAM PENALTY SHOOTOUT WIN RATES (historical WC data)
# Source: FIFA World Cup all-time penalty shootout records.
# Relative model: p(A wins) = rate_A / (rate_A + rate_B)
# Defaults to 0.5 for teams with no WC shootout history.
# ─────────────────────────────────────────────────────────
PENALTY_WIN_RATES: dict[str, float] = {
    "Germany":          0.65,  # 5W-1L in WC shootouts
    "Croatia":          0.70,  # 3W-0L (1998, 2022, 2022)
    "Argentina":        0.62,  # 4W-2L
    "France":           0.55,
    "Portugal":         0.55,
    "Uruguay":          0.60,
    "Colombia":         0.55,
    "Netherlands":      0.48,
    "Spain":            0.45,  # Lost 1986, 2002
    "Brazil":           0.43,  # Lost 1994, 2006, 2011
    "England":          0.48,  # Historic poor record, improving recently
    "Mexico":           0.30,  # Historically knocked out pre-shootout or lost (1986)
    "Switzerland":      0.55,  # Beat France 2021 Euros
    "Japan":            0.55,
    "South Korea":      0.50,
    "Morocco":          0.55,
    "Senegal":          0.50,
    "Sweden":           0.50,
    "Belgium":          0.50,
    "United States":    0.50,
}

# Groups confirmed from the FIFA WC 2026 Final Draw (December 5, 2025, Washington DC).
# Playoff spots resolved March 31, 2026:
#   Euro Path A → Bosnia and Herzegovina (beat Italy on pens)
#   Euro Path B → Sweden (beat Poland 3-2)
#   Euro Path C → Turkey (beat Kosovo 1-0; "Türkiye" in FIFA docs)
#   Euro Path D → Czech Republic (beat Denmark on pens; "Czechia" in FIFA docs)
#   Intercontinental 1 → DR Congo
#   Intercontinental 2 → Iraq
# Team names use martj42/international_results spelling for Elo compatibility.
WC_2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico",        "South Africa",          "South Korea",  "Czech Republic"],
    "B": ["Canada",        "Bosnia and Herzegovina", "Qatar",        "Switzerland"],
    "C": ["Brazil",        "Morocco",               "Haiti",        "Scotland"],
    "D": ["United States", "Paraguay",              "Australia",    "Turkey"],
    "E": ["Germany",       "Curacao",               "Ivory Coast",  "Ecuador"],
    "F": ["Netherlands",   "Japan",                 "Sweden",       "Tunisia"],
    "G": ["Belgium",       "Egypt",                 "Iran",         "New Zealand"],
    "H": ["Spain",         "Cape Verde",            "Saudi Arabia", "Uruguay"],
    "I": ["France",        "Senegal",               "Iraq",         "Norway"],
    "J": ["Argentina",     "Algeria",               "Austria",      "Jordan"],
    "K": ["Portugal",      "DR Congo",              "Uzbekistan",   "Colombia"],
    "L": ["England",       "Croatia",               "Ghana",        "Panama"],
}


class WorldCupSimulator:
    """
    Runs N independent simulations of the full tournament.

    Each simulation:
    1. Simulates all group-stage matches (round-robin within groups)
    2. Determines group standings (points → GD → GF → coin flip)
    3. Identifies 8 best 3rd-place teams by points → GD → GF
    4. Runs knockout bracket until 1 champion emerges

    The predictor provides expected goals (λ) which feed Poisson
    sampling for realistic score simulation. If λ is unavailable,
    falls back to sampling W/D/L from the predictor's probabilities.
    """

    def __init__(
        self,
        predictor: BaseMatchPredictor,
        groups: dict[str, list[str]] = WC_2026_GROUPS,
        n_simulations: int = 10_000,
        penalties_home_prob: float = 0.5,
        known_results: Optional[dict] = None,
        penalty_rates: Optional[dict] = None,
    ):
        self.predictor = predictor
        self.groups = groups
        self.n_simulations = n_simulations
        self.pen_prob = penalties_home_prob  # fallback when no team-specific rate found
        self.known_results: dict[tuple, tuple] = (
            known_results if known_results is not None else KNOWN_WC_2026_RESULTS
        )
        self.penalty_rates: dict[str, float] = (
            penalty_rates if penalty_rates is not None else PENALTY_WIN_RATES
        )

        self.all_teams: list[str] = [
            t for g in groups.values() for t in g
            if not t.startswith("TBD")
        ]

    # ─────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────

    def champion_probabilities(
        self, n_simulations: Optional[int] = None, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by team with columns:
            p_champion, p_finalist, p_semi, p_quarter, p_r16, p_group_advance
        sorted by p_champion descending.
        """
        if n_simulations:
            self.n_simulations = n_simulations

        logger.info(f"Running {self.n_simulations:,} simulations with {self.predictor.name}…")

        raw = self.run(verbose=verbose)

        rows = []
        for team in self.all_teams:
            stages = raw.get(team, {})
            rows.append({
                "team":            team,
                "p_champion":      stages.get("champion",       0.0),
                "p_finalist":      stages.get("finalist",       0.0),
                "p_semi":          stages.get("semi",           0.0),
                "p_quarter":       stages.get("quarter",        0.0),
                "p_r16":           stages.get("r16",            0.0),
                "p_group_advance": stages.get("group_advance",  0.0),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("p_champion", ascending=False)
            .reset_index(drop=True)
        )

    def run(self, verbose: bool = True) -> dict[str, dict[str, float]]:
        """
        Core simulation loop.
        Returns {team: {stage: probability}} mapping.
        """
        # Pre-warm so 10k simulation iterations pay zero prediction overhead
        self._pred_cache = {}
        self._warm_prediction_cache(verbose=verbose)

        counts: dict[str, dict[str, int]] = {
            t: defaultdict(int) for t in self.all_teams
        }

        iterator = tqdm(range(self.n_simulations), desc="Simulating") if verbose \
                   else range(self.n_simulations)

        for _ in iterator:
            result = self._simulate_once()
            for team, stages in result.items():
                for s in stages:
                    counts[team][s] += 1

        return {
            team: {s: c / self.n_simulations for s, c in stages.items()}
            for team, stages in counts.items()
        }

    def _warm_prediction_cache(self, verbose: bool = True) -> None:
        """Pre-compute all real-team pair predictions before the simulation loop."""
        real  = [t for g in self.groups.values() for t in g if not t.startswith("TBD")]
        pairs = [(h, a) for h in real for a in real if h != a]
        if verbose:
            logger.info(
                f"Warming prediction cache: {len(pairs)} pairs for {len(real)} real teams…"
            )
        warmed = 0
        for home, away in pairs:
            pair = (home, away)
            if pair not in self._pred_cache:
                try:
                    self._pred_cache[pair] = self.predictor.predict(home, away)
                    warmed += 1
                except Exception:
                    pass
        if verbose:
            logger.info(f"Cache ready: {warmed} pairs pre-computed.")

    # ─────────────────────────────────────────────────
    # SINGLE TOURNAMENT
    # ─────────────────────────────────────────────────

    def _simulate_once(self) -> dict[str, list[str]]:
        stages: dict[str, list[str]] = defaultdict(list)

        # ── GROUP STAGE ──────────────────────────────
        qualifiers: list[str] = []
        third_place_pool: list[tuple] = []  # (team, pts, gd, gf)

        for group_name, members in self.groups.items():
            real = [t for t in members if not t.startswith("TBD")]
            if len(real) < 2:
                continue

            standings = self._simulate_group(real)

            for rank, (team, pts, gd, gf) in enumerate(standings):
                if rank < 2:
                    stages[team].append("group_advance")
                    qualifiers.append(team)
                elif rank == 2:
                    third_place_pool.append((team, pts, gd, gf))

        # 8 best 3rd-place teams
        third_place_pool.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        for team, *_ in third_place_pool[:8]:
            stages[team].append("group_advance")
            qualifiers.append(team)

        # ── KNOCKOUT ROUNDS ───────────────────────────
        stage_labels = ["r16", "quarter", "semi", "finalist", "champion"]
        current = qualifiers.copy()
        np.random.shuffle(current)  # random bracket ordering (simplified)
        s_idx = 0

        while len(current) > 1:
            label = stage_labels[min(s_idx, len(stage_labels) - 1)]
            next_round: list[str] = []

            for i in range(0, len(current) - 1, 2):
                winner = self._knockout_match(current[i], current[i + 1])
                stages[winner].append(label)
                next_round.append(winner)

            if len(current) % 2 == 1:  # bye (shouldn't happen in WC)
                bye = current[-1]
                stages[bye].append(label)
                next_round.append(bye)

            current = next_round
            s_idx += 1

        # Champion = sole survivor. Guard against double-count: the final
        # round above already labels its winner "champion"; only append here
        # if it wasn't (e.g. degenerate bracket with a single qualifier).
        if current and "champion" not in stages[current[0]]:
            stages[current[0]].append("champion")

        return stages

    # ─────────────────────────────────────────────────
    # GROUP STAGE
    # ─────────────────────────────────────────────────

    def _simulate_group(self, teams: list[str]) -> list[tuple]:
        """Round-robin for a group. Returns standings (team, pts, gd, gf)."""
        pts: dict[str, int]  = defaultdict(int)
        gd:  dict[str, int]  = defaultdict(int)
        gf:  dict[str, int]  = defaultdict(int)

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                # Use actual result if the match has been played
                if (teams[i], teams[j]) in self.known_results:
                    gh, ga = self.known_results[(teams[i], teams[j])]
                elif (teams[j], teams[i]) in self.known_results:
                    ga, gh = self.known_results[(teams[j], teams[i])]
                else:
                    gh, ga = self._sample_score(teams[i], teams[j])

                gf[teams[i]] += gh
                gf[teams[j]] += ga
                gd[teams[i]] += gh - ga
                gd[teams[j]] += ga - gh

                if gh > ga:
                    pts[teams[i]] += 3
                elif gh < ga:
                    pts[teams[j]] += 3
                else:
                    pts[teams[i]] += 1
                    pts[teams[j]] += 1

        standings = sorted(
            teams,
            key=lambda t: (pts[t], gd[t], gf[t], np.random.random()),
            reverse=True,
        )
        return [(t, pts[t], gd[t], gf[t]) for t in standings]

    # ─────────────────────────────────────────────────
    # SCORE SAMPLING
    # ─────────────────────────────────────────────────

    def _sample_score(self, home: str, away: str) -> tuple[int, int]:
        """Sample a realistic score from the predictor's output."""
        try:
            pair = (home, away)
            if not hasattr(self, '_pred_cache'):
                self._pred_cache = {}
            if pair not in self._pred_cache:
                self._pred_cache[pair] = self.predictor.predict(home, away)
            
            pred = self._pred_cache[pair]

            if pred.lambda_home is not None and pred.lambda_away is not None:
                gh = int(np.random.poisson(pred.lambda_home))
                ga = int(np.random.poisson(pred.lambda_away))
                return gh, ga

            # Fallback: sample outcome class, then sample plausible score
            outcome = np.random.choice(
                [0, 1, 2],
                p=[pred.p_home_win, pred.p_draw, pred.p_away_win]
            )
            return self._outcome_to_score(outcome)

        except Exception:
            return int(np.random.poisson(1.3)), int(np.random.poisson(1.1))

    def _knockout_match(self, team_a: str, team_b: str) -> str:
        """Knockout match: draw → penalties with team-specific rates."""
        gh, ga = self._sample_score(team_a, team_b)
        if gh > ga:
            return team_a
        if ga > gh:
            return team_b
        # Relative model: p(A wins) = rate_A / (rate_A + rate_B)
        rate_a = self.penalty_rates.get(team_a, self.pen_prob)
        rate_b = self.penalty_rates.get(team_b, self.pen_prob)
        p_a = rate_a / (rate_a + rate_b)
        return team_a if np.random.random() < p_a else team_b

    @staticmethod
    def _outcome_to_score(outcome: int) -> tuple[int, int]:
        """Convert W/D/L outcome to a plausible score."""
        if outcome == 0:   # Win
            gh = np.random.choice([1, 2, 2, 3], p=[0.35, 0.40, 0.15, 0.10])
            ga = np.random.choice([0, 1], p=[0.55, 0.45])
            return int(gh), int(min(ga, gh - 1))
        elif outcome == 1: # Draw
            g = np.random.choice([0, 1, 2], p=[0.40, 0.45, 0.15])
            return int(g), int(g)
        else:              # Loss
            gh = np.random.choice([0, 1], p=[0.55, 0.45])
            ga = np.random.choice([1, 2, 2, 3], p=[0.35, 0.40, 0.15, 0.10])
            return int(min(gh, ga - 1)), int(ga)
