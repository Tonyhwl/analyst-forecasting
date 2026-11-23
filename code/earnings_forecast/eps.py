"""EPS forecasts and beat probabilities for the January cohorts."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from earnings_forecast.config import (
    ACTUAL, CONSENSUS, DATA_DIR, DATA_VINTAGE, EPS_FEATURES,
)
from earnings_forecast.data import ReleaseData, Roster
from earnings_forecast.features import EpsPanel
from earnings_forecast.inference import relative_reduction
from earnings_forecast.models import AnnualEpsForecast, BeatModel

FORECAST = 'expanded_day1'
RECENT4 = 'recent4_day1'
KEYS = ['cik', 'fiscal_quarter', 'announcement_date']


class EpsStudy:
    """Score the boosted EPS forecast against consensus and recent surprises."""

    def __init__(self, period, output_dir, data_dir=DATA_DIR):
        self.period = period
        self.output_dir = output_dir
        self.roster = Roster(period.cohort, data_dir)
        self.data = ReleaseData(self.roster, data_dir)
        self.model = AnnualEpsForecast(ACTUAL, CONSENSUS, EPS_FEATURES)

    @property
    def forecast_file(self):
        return self.output_dir / (
            f'january_cohort_{self.period.name}_{DATA_VINTAGE}.csv')

    def panel(self):
        return EpsPanel.build(self.data.eps(self.period.eps_end))

    def run(self):
        panel = self.panel()
        with threadpool_limits(limits=1):
            panel[FORECAST] = self.model.predict(panel, self.period.start)
        scored = panel.loc[panel.as_of.ge(self.period.start)].copy()
        assert scored[FORECAST].notna().all()
        scored = scored.rename(columns={'recent4_bias_eps': RECENT4})
        scored.to_csv(self.forecast_file, index=False)
        coverage = self.roster.coverage(scored.groupby('cik').size())
        coverage.to_csv(self.output_dir / (
            f'january_coverage_{self.period.name}_{DATA_VINTAGE}.csv'),
            index=False)
        return scored


class BeatStudy:
    """Beat probabilities with and without the boosted EPS forecast gap."""

    RATE = 'historical_beat_rate'
    SEVEN = 'seven_feature_probability'
    GAP = 'gap_probability'
    COMBINED = 'seven_plus_gap_probability'
    MATCHED = 'seven_matched_training_probability'

    def __init__(self, period, output_dir, data_dir=DATA_DIR):
        self.period = period
        self.output_dir = output_dir
        self.eps = EpsStudy(period, output_dir, data_dir)

    def panel(self):
        panel = self.eps.panel()
        panel['beat'] = panel[ACTUAL].gt(panel[CONSENSUS]).astype(int)
        return panel

    def probabilities(self, panel):
        """Seven-input probabilities, checked against the saved EPS forecasts."""
        panel = panel.copy()
        panel[self.SEVEN] = BeatModel(EPS_FEATURES).walk_forward(panel)
        saved = pd.read_csv(self.eps.forecast_file, dtype={'cik': str},
                            parse_dates=['announcement_date'],
                            usecols=KEYS + [ACTUAL, CONSENSUS, FORECAST])
        scored = panel.loc[panel.as_of.ge(self.period.start)]
        rows = scored.merge(saved, how='outer', on=KEYS, validate='one_to_one',
                            indicator=True, suffixes=('', '_saved'))
        if not rows['_merge'].eq('both').all():
            raise ValueError('Scored releases differ from the EPS evaluation')
        for col in [CONSENSUS, ACTUAL]:
            np.testing.assert_allclose(rows[col], rows[col + '_saved'])
        if not rows[[self.SEVEN, self.RATE]].apply(
                lambda col: col.between(0, 1).all()).all():
            raise ValueError('Missing or invalid probabilities')
        rows['ticker'] = rows.cik.map(self.eps.roster.tickers())
        rows['forecast_category'] = BeatModel.category(rows[self.SEVEN])
        return rows[KEYS + ['ticker', 'as_of', CONSENSUS, FORECAST, ACTUAL,
                            self.SEVEN, self.RATE, 'beat', 'forecast_category']]

    def with_gap(self, panel):
        """Add the gap-only, combined and matched-training probabilities."""
        boosted = 'boosted_forecast'
        model = AnnualEpsForecast(ACTUAL, CONSENSUS, EPS_FEATURES)
        with threadpool_limits(limits=1):
            eps = model.predict_all(panel, boosted)
        eps = eps.rename(columns={ACTUAL: 'actual'})
        rows = BeatModel.add_forecast_gap(eps, boosted, CONSENSUS).dropna(
            subset=['forecast_gap']).copy()
        for name, features in [
            (self.GAP, ['forecast_gap']),
            (self.COMBINED, EPS_FEATURES + ['forecast_gap']),
            (self.MATCHED, EPS_FEATURES),
        ]:
            rows[name] = BeatModel(features).walk_forward(rows)
        saved = pd.read_csv(self.probability_file, dtype={'cik': str},
                            parse_dates=['announcement_date'])
        scored = rows.loc[rows.as_of.ge(self.period.start)]
        extra = ['forecast_gap', boosted, self.GAP, self.COMBINED, self.MATCHED]
        paired = saved.merge(scored[KEYS + extra], on=KEYS, how='left',
                             validate='one_to_one')
        eligible = paired.dropna(subset=['forecast_gap']).copy()
        if len(eligible) != len(scored):
            raise ValueError('Forecast releases differ from the saved archive')
        if not eligible[[self.GAP, self.COMBINED, self.MATCHED]].apply(
                lambda col: col.between(0, 1).all()).all():
            raise ValueError('Missing or invalid probabilities')
        np.testing.assert_allclose(eligible[boosted], eligible[FORECAST],
                                   atol=1e-10)
        return eligible.drop(columns=boosted), len(saved)

    @property
    def probability_file(self):
        return self.output_dir / f'january_beat_{self.period.name}.csv'

    @staticmethod
    def summary(rows, forecast, benchmark):
        loss = rows[[forecast, benchmark]].sub(rows.beat, axis=0).pow(2)
        model, base = loss.mean()
        gain, low, high = relative_reduction(
            loss[benchmark], loss[forecast],
            rows.announcement_date.dt.to_period('Q'))
        return dict(issuers=rows.cik.nunique(), releases=len(rows),
                    mean_probability_pct=100 * rows[forecast].mean(),
                    beat_pct=100 * rows.beat.mean(), brier_model=model,
                    brier_history=base, improvement_pct=gain,
                    lower_pct=low, upper_pct=high)

    COMPARISONS = [
        (SEVEN, RATE), (MATCHED, RATE), (GAP, RATE), (COMBINED, RATE),
        (GAP, SEVEN), (COMBINED, SEVEN), (COMBINED, MATCHED), (COMBINED, GAP),
    ]

    def run(self):
        panel = self.panel()
        rows = self.probabilities(panel).sort_values(['announcement_date', 'cik'])
        rows.to_csv(self.probability_file, index=False)
        seven = dict(period=self.period.name,
                     **self.summary(rows, self.SEVEN, self.RATE))
        gap_rows, total = self.with_gap(panel)
        gap_rows.to_csv(self.output_dir / (
            f'january_beat_gap_{self.period.name}.csv'), index=False)
        comparisons = [
            dict(period=self.period.name, model=model, benchmark=benchmark,
                 available=total, **self.summary(gap_rows, model, benchmark))
            for model, benchmark in self.COMPARISONS
        ]
        return seven, comparisons
