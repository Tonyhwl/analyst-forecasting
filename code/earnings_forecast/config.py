"""Paths, evaluation periods and the fixed model settings."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'

ROSTER_FILE = 'sp500_january_2017_2021_members.csv'
EPS_FILE = 'sp500_january_cohort_eps_2026-11-22.csv'
REVENUE_EBIT_FILE = 'sp500_january_cohort_revenue_ebit_2026-11-22.csv'
DATA_VINTAGE = '2026-11-22'
LAST_SCORED_CUTOFF = pd.Timestamp('2026-11-19')

# Column names shared across the EPS studies.
ACTUAL = 'normalised_actual_eps'
CONSENSUS = 'consensus_eps_1d'
EPS_FEATURES = ['consensus_level', 'last_surprise', 'median_four',
                'median_twelve', 'beat_rate', 'seasonal_surprise',
                'eps_change']

# Median gradient boosting, fixed during development and never tuned on
# later data.
BOOSTING_SETTINGS = dict(
    loss='quantile', quantile=0.5, max_iter=60, learning_rate=0.05,
    max_leaf_nodes=3, min_samples_leaf=30, l2_regularization=1.0,
    early_stopping=False, random_state=42,
)


@dataclass(frozen=True)
class Period:
    """An evaluation period scored with the roster fixed at its start."""

    name: str
    first_year: int
    last_year: int
    eps_end: pd.Timestamp  # EPS releases on or after this date are unused

    @property
    def cohort(self):
        return self.first_year

    @property
    def start(self):
        return pd.Timestamp(self.first_year, 1, 1)

    @property
    def years(self):
        return range(self.first_year, self.last_year + 1)


DEVELOPMENT = Period('development', 2017, 2020, pd.Timestamp('2021-01-01'))
LATER = Period('later', 2021, 2026, pd.Timestamp('2026-11-22'))
PERIODS = (DEVELOPMENT, LATER)
