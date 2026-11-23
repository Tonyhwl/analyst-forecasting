"""Revenue and implied EBIT margin forecasts for the January cohorts."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from earnings_forecast.config import DATA_DIR
from earnings_forecast.data import ReleaseData, Roster
from earnings_forecast.inference import bootstrap_relative_reduction
from earnings_forecast.models import SurpriseForecast

KEYS = ['cik', 'quarter', 'date', 'metric']


def revenue_scale(rows):
    """Revenue corrections are a percentage of consensus; margin, points."""
    return np.where(rows.metric.eq('Revenue'), rows.consensus / 100, 1)


class RevenueMarginStudy:
    """Annual walk-forward revenue and margin forecasts for one period."""

    def __init__(self, period, output_dir, data_dir=DATA_DIR):
        self.period = period
        self.output_dir = output_dir
        self.roster = Roster(period.cohort, data_dir)
        self.data = ReleaseData(self.roster, data_dir)

    @property
    def forecast_file(self):
        return self.output_dir / (
            f'sp500_revenue_margin_{self.period.name}_forecasts.csv')

    def observations(self):
        return self.data.revenue_and_margin(self.period.last_year)

    def forecasts(self):
        raw = self.observations()
        with threadpool_limits(1):
            scores = SurpriseForecast.evaluate(raw, self.period.years)
        scores = scores.merge(raw, on=KEYS, validate='one_to_one')
        scale = revenue_scale(scores)
        scores['model_forecast'] = (
            scores.consensus + scale * scores.predicted_correction)
        scores['recent4_forecast'] = scores.consensus + scale * scores.recent4
        assert np.allclose(abs(scores.actual - scores.model_forecast) / scale,
                           scores.model_error)
        assert scores.date.dt.year.between(
            self.period.first_year, self.period.last_year).all()
        return scores

    @staticmethod
    def summary(metric, period, rows):
        errors = rows[['consensus_error', 'recent4_error', 'model_error']].mean()
        return dict(
            target=metric, period=period, companies=rows.cik.nunique(),
            releases=len(rows), consensus=errors.consensus_error,
            recent4=errors.recent4_error, model=errors.model_error,
            gain_vs_consensus=100 * (1 - errors.model_error
                                     / errors.consensus_error),
            gain_vs_recent4=100 * (1 - errors.model_error
                                   / errors.recent4_error))

    def run(self):
        scores = self.forecasts()
        rows, intervals = [], {}
        span = f'{self.period.first_year}-{self.period.last_year}'
        for metric, group in scores.groupby('metric', sort=False):
            for year, part in group.groupby(group.date.dt.year):
                rows.append(self.summary(metric, year, part))
            rows.append(self.summary(metric, span, group))
            totals = group.groupby(group.date.dt.to_period('Q'))[
                ['recent4_error', 'model_error']].sum().to_numpy()
            assert len(totals) == 4 * len(self.period.years)
            intervals[metric] = bootstrap_relative_reduction(totals)
        name = self.period.name
        pd.DataFrame(rows).to_csv(
            self.output_dir / f'sp500_revenue_margin_{name}_summary.csv',
            index=False)
        scores.to_csv(self.forecast_file, index=False)
        return intervals


class CompanyBreadth:
    """How many companies improved, and the pooled gain with its interval."""

    def __init__(self, period, output_dir, data_dir=DATA_DIR):
        self.study = RevenueMarginStudy(period, output_dir, data_dir)
        self.names = Roster(period.cohort, data_dir).members.groupby(
            'cik').company.first()

    def run(self):
        prefix = f'sp500_revenue_margin_{self.study.period.name}'
        data = pd.read_csv(self.study.forecast_file, dtype={'cik': str},
                           parse_dates=['date'])
        assert data.cik.isin(self.names.index).all()
        assert not data.duplicated(['metric', 'cik', 'quarter']).any()
        companies, breadth = [], []
        for target, group in data.groupby('metric'):
            for base in ['consensus_error', 'recent4_error']:
                firms = group.groupby('cik').agg(
                    releases=('date', 'size'), baseline_mae=(base, 'mean'),
                    model_mae=('model_error', 'mean')).join(self.names)
                change = firms.baseline_mae - firms.model_mae
                firms['gain_pct'] = 100 * change / firms.baseline_mae.where(
                    firms.baseline_mae > 0)
                firms['win'] = change.gt(0)
                firms['target'], firms['benchmark'] = target, base
                companies.append(firms.reset_index())
                totals = group.groupby(group.date.dt.to_period('Q'))[
                    [base, 'model_error']].sum().to_numpy()
                lo, hi = bootstrap_relative_reduction(totals)
                breadth.append(dict(
                    target=target, benchmark=base, companies=len(firms),
                    pooled_gain=100 * (1 - group.model_error.sum()
                                       / group[base].sum()),
                    median_firm_gain=firms.gain_pct.median(),
                    wins=int(firms.win.sum()), win_pct=100 * firms.win.mean(),
                    ties=int(change.eq(0).sum()),
                    undefined_gains=int(firms.gain_pct.isna().sum()),
                    ci_low=lo, ci_high=hi))
        out = self.study.output_dir
        pd.concat(companies).to_csv(out / f'{prefix}_companies.csv',
                                    index=False)
        pd.DataFrame(breadth).to_csv(out / f'{prefix}_breadth.csv',
                                     index=False)
