"""Revenue and EBIT margin miss risk: development and the frozen later test."""

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from earnings_forecast.config import (
    DATA_DIR, DEVELOPMENT, LAST_SCORED_CUTOFF, Period,
)
from earnings_forecast.data import Roster
from earnings_forecast.inference import relative_reduction
from earnings_forecast.models import MissRiskModel, SurpriseForecast
from earnings_forecast.revenue_margin import KEYS, RevenueMarginStudy

FIRST_FORECAST_YEAR = 2014


def probability_bands(rows):
    """Observed miss rates in 10-point bands, split by risk versus history."""
    rows = rows.copy()
    rows['band_start_pct'] = np.searchsorted(
        np.arange(1, 10) / 10, rows.model, side='right') * 10
    rows['direction'] = np.select(
        [rows.model > rows.history_rate, rows.model < rows.history_rate],
        ['above', 'below'], default='equal')
    pieces = []
    for direction in ('all', 'above', 'below', 'equal'):
        subset = rows if direction == 'all' else rows.loc[
            rows.direction.eq(direction)]
        bands = subset.groupby('band_start_pct').agg(
            releases=('miss', 'size'), firms=('cik', 'nunique'),
            misses=('miss', 'sum'), mean_model=('model', 'mean'),
            observed=('miss', 'mean'), history=('history_rate', 'mean'))
        bands = bands.reindex(range(0, 100, 10))
        for name in ['releases', 'firms', 'misses']:
            bands[name] = bands[name].fillna(0).astype(int)
        pieces.append(bands.reset_index().assign(direction=direction))
    result = pd.concat(pieces, ignore_index=True)
    assert result.loc[result.direction.eq('all'), 'releases'].sum() == len(rows)
    assert result.loc[result.direction.ne('all'), 'releases'].sum() == len(rows)
    return result


class MissForecasts:
    """Rebuild point forecasts from 2014 and attach each firm's miss history."""

    def __init__(self, period, output_dir, data_dir=DATA_DIR):
        self.period = period
        self.study = RevenueMarginStudy(period, output_dir, data_dir)

    def build(self):
        raw = self.study.observations()
        assert raw.date.max() < pd.Timestamp(self.period.last_year + 1, 1, 1)
        years = range(FIRST_FORECAST_YEAR, self.period.last_year + 1)
        with threadpool_limits(1):
            forecasts = SurpriseForecast.evaluate(raw, years)
        raw = raw.sort_values('date').copy()
        raw['miss'] = raw.actual.lt(raw.consensus).astype(int)
        forecasts = forecasts.merge(raw, on=KEYS, validate='one_to_one')
        # A firm's miss rate can use all settled actuals, including those
        # before the first model forecast.
        grouped = raw.groupby(['metric', 'cik'])
        raw['history_rate'] = grouped.miss.cumsum() / (grouped.cumcount() + 1)
        history = raw.rename(columns={'date': 'rate_date'})
        forecasts['cutoff'] = forecasts.date - pd.Timedelta(days=1)
        forecasts = pd.merge_asof(
            forecasts.sort_values('cutoff'),
            history[['metric', 'cik', 'rate_date', 'history_rate']],
            by=['metric', 'cik'], left_on='cutoff', right_on='rate_date',
            allow_exact_matches=False)
        assert forecasts.rate_date.lt(forecasts.cutoff).all()
        self.check_saved(forecasts)
        return forecasts

    def check_saved(self, forecasts):
        """The rebuilt forecasts must match the saved period forecasts."""
        keys = ['metric', 'cik', 'quarter', 'date']
        saved = pd.read_csv(
            self.study.forecast_file, dtype={'cik': str}, parse_dates=['date'],
            float_precision='round_trip').set_index(keys).sort_index()
        rebuilt = forecasts.loc[
            forecasts.date.dt.year.ge(self.period.first_year)
        ].set_index(keys).sort_index()
        pd.testing.assert_index_equal(saved.index, rebuilt.index)
        for col in ['actual', 'consensus', 'predicted_correction',
                    'model_error']:
            np.testing.assert_allclose(rebuilt[col], saved[col], rtol=0,
                                       atol=1e-10)
        scale = np.where(rebuilt.index.get_level_values('metric') == 'Revenue',
                         rebuilt.consensus / 100, 1)
        np.testing.assert_allclose(
            rebuilt.consensus + scale * rebuilt.predicted_correction,
            saved.model_forecast, rtol=0, atol=1e-10)


class DevelopmentMissRisk:
    """Annual miss probabilities for 2017 to 2020."""

    def __init__(self, output_dir, data_dir=DATA_DIR):
        self.output_dir = output_dir
        self.forecasts = MissForecasts(DEVELOPMENT, output_dir, data_dir)

    def run(self):
        forecasts = self.forecasts.build()
        summaries, predictions, bands = [], [], []
        for metric, group in forecasts.groupby('metric'):
            test = MissRiskModel(group, DEVELOPMENT.years).evaluate()
            predictions.append(test[[
                'metric', 'cik', 'quarter', 'date', 'year', 'training', 'miss',
                'scaled_gap', 'history_rate', 'rate_only', 'model']])
            for method in ['history_rate', 'rate_only', 'model']:
                p = test[method]
                assert p.between(0, 1).all()
                bins = np.searchsorted(np.arange(1, 10) / 10, p,
                                       side='right') * 10
                band = test.groupby(bins, observed=True).agg(
                    releases=('miss', 'size'), predicted=(method, 'mean'),
                    observed=('miss', 'mean'))
                bands.append(band.rename_axis('band_start_pct').reset_index()
                             .assign(metric=metric, method=method))
                for period, sample in [*test.groupby('year'), ('Pooled', test)]:
                    probability = sample[method]
                    loss = (probability - sample.miss).pow(2)
                    base = (sample.history_rate - sample.miss).pow(2)
                    gain, low, high = relative_reduction(
                        base, loss, sample.date.dt.to_period('Q'))
                    called = probability.ge(.5)
                    summaries.append(dict(
                        metric=metric, period=period, method=method,
                        releases=len(sample), firms=sample.cik.nunique(),
                        misses=sample.miss.sum(),
                        mean_probability=probability.mean(),
                        brier=loss.mean(), gain=gain, ci_low=low, ci_high=high,
                        called=called.sum(),
                        correct=sample.loc[called, 'miss'].sum()))
        for suffix, table in [('summary', pd.DataFrame(summaries)),
                              ('calibration', pd.concat(bands)),
                              ('predictions', pd.concat(predictions))]:
            table.to_csv(self.output_dir / (
                f'walk_forward_development_miss_{suffix}.csv'), index=False)


class DevelopmentCalibration:
    """Development miss rates by 10-point band, pooled and by company."""

    def __init__(self, output_dir, data_dir=DATA_DIR):
        self.output = output_dir
        roster = Roster(DEVELOPMENT.cohort, data_dir).members
        self.companies = roster.groupby('cik').agg(
            tickers=('ticker', '/'.join), company=('company', 'first'))
        keys = ['metric', 'cik', 'quarter', 'date']
        predictions = pd.read_csv(
            output_dir / 'walk_forward_development_miss_predictions.csv',
            dtype={'cik': str})
        actuals = pd.read_csv(
            output_dir / 'sp500_revenue_margin_development_forecasts.csv',
            dtype={'cik': str})
        rows = predictions.merge(actuals[keys + ['actual', 'consensus']],
                                 how='left', on=keys, validate='one_to_one')
        assert rows.year.between(2017, 2020).all()
        assert rows.cik.isin(self.companies.index).all()
        assert rows[['actual', 'consensus', 'history_rate',
                     'model']].notna().all().all()
        assert rows.model.between(0, 1).all()
        assert rows.miss.eq(rows.actual.lt(rows.consensus)).all()
        rows['band_start_pct'] = 10 * np.searchsorted(
            np.arange(1, 10) / 10, rows.model, side='right')
        rows['beat'] = rows.actual.gt(rows.consensus)
        rows['match'] = rows.actual.eq(rows.consensus)
        assert (rows.miss + rows.beat.astype(int)
                + rows.match.astype(int)).eq(1).all()
        self.rows = rows

    @staticmethod
    def summarise(rows, groups):
        table = rows.groupby(groups).agg(
            releases=('miss', 'size'), firms=('cik', 'nunique'),
            misses=('miss', 'sum'), beats=('beat', 'sum'),
            matches=('match', 'sum'), mean_predicted_pct=('model', 'mean'),
            actual_miss_pct=('miss', 'mean'),
            historical_miss_pct=('history_rate', 'mean'),
            actual_beat_pct=('beat', 'mean'))
        rates = [col for col in table if col.endswith('_pct')]
        table[rates] *= 100
        table['miss_minus_history_pp'] = (
            table.actual_miss_pct - table.historical_miss_pct)
        return table

    def run(self):
        rows = self.rows
        bands = range(0, 100, 10)
        metrics = sorted(rows.metric.unique())
        counts = ['releases', 'firms', 'misses', 'beats', 'matches']
        for name, groups, levels in [
                ('bands', ['metric', 'band_start_pct'], [metrics, bands]),
                ('company_bands', ['metric', 'cik', 'band_start_pct'],
                 [metrics, self.companies.index, bands])]:
            index = pd.MultiIndex.from_product(levels, names=groups)
            table = self.summarise(rows, groups).reindex(index)
            table[counts] = table[counts].fillna(0).astype(int)
            assert table.releases.sum() == len(rows)
            table = table.reset_index()
            if 'cik' in groups:
                table = table.merge(self.companies, on='cik',
                                    validate='many_to_one')
            table.to_csv(self.output / f'development_miss_{name}_10pp.csv',
                         index=False, float_format='%.4f')
        split = pd.concat([probability_bands(group).assign(metric=metric)
                           for metric, group in rows.groupby('metric')],
                          ignore_index=True)
        split.to_csv(self.output / 'development_miss_band_history_split.csv',
                     index=False)


class FrozenMissRiskTest:
    """The single 2021 to 2026 evaluation frozen in EVALUATION_PROTOCOL.md."""

    PREFIX = 'later_miss_risk_'
    GAP_LABELS = ['below -1', '-1 to -0.25', 'close', '0.25 to 1', '1 to 2',
                  'above 2']

    def __init__(self, period: Period, output_dir, data_dir=DATA_DIR):
        self.period = period
        self.output_dir = output_dir
        self.forecasts = MissForecasts(period, output_dir, data_dir)

    @staticmethod
    def top_decile(rows):
        """Miss rate among the riskiest tenth: model versus history."""
        # Round to the nearest release; order equal risks by earlier date,
        # then ascending CIK.
        count = max(1, int(np.floor(len(rows) / 10 + .5)))
        quarters = rows.date.dt.to_period('Q')
        influence = pd.Series(0., index=rows.index)
        groups = {}
        for column, sign in [('model', 1), ('history_rate', -1)]:
            chosen = rows.sort_values([column, 'date', 'cik'],
                                      ascending=[False, True, True]).head(count)
            rate = chosen.miss.mean()
            groups[column] = rate
            influence.loc[chosen.index] += sign * (chosen.miss - rate) / count
        difference = groups['model'] - groups['history_rate']
        totals = influence.groupby(quarters).sum()
        clusters = len(totals)
        half = t.ppf(.975, clusters - 1) * np.sqrt(
            clusters / (clusters - 1) * totals.pow(2).sum())
        return dict(top_n=count, top_model=groups['model'],
                    top_history=groups['history_rate'],
                    difference_pp=100 * difference,
                    lower_pp=100 * (difference - half),
                    upper_pp=100 * (difference + half),
                    passes=bool(difference - half > 0), clusters=clusters)

    @classmethod
    def summarise(cls, rows):
        base = (rows.history_rate - rows.miss).pow(2)
        loss = (rows.model - rows.miss).pow(2)
        gain, low, high = relative_reduction(base, loss,
                                             rows.date.dt.to_period('Q'))
        return dict(releases=len(rows), companies=rows.cik.nunique(),
                    miss_rate=rows.miss.mean(), mean_model=rows.model.mean(),
                    **cls.top_decile(rows), history_brier=base.mean(),
                    model_brier=loss.mean(), brier_gain_pct=gain,
                    brier_low=low, brier_high=high,
                    model_auc=roc_auc_score(rows.miss, rows.model),
                    history_auc=roc_auc_score(rows.miss, rows.history_rate))

    @classmethod
    def gap_table(cls, rows):
        g = rows.scaled_gap
        labels = cls.GAP_LABELS
        band = np.select([g.lt(-1), g.lt(-.25), g.le(.25), g.le(1), g.le(2)],
                         labels[:-1], default=labels[-1])
        result = rows.groupby(band).agg(
            releases=('miss', 'size'), observed=('miss', 'mean'),
            mean_model=('model', 'mean')).reindex(labels)
        result.index.name = 'gap_band'
        result['whole_sample'] = rows.miss.mean()
        holds = (result.loc['close', 'observed'] > rows.miss.mean()
                 and (result.observed.iloc[3:] < rows.miss.mean()).all())
        return result.reset_index(), bool(holds)

    def run(self):
        predictions = self.output_dir / (self.PREFIX + 'predictions.csv')
        if predictions.exists():
            raise FileExistsError('The later predictions have already been '
                                  'saved; write to a new directory.')
        forecasts = self.forecasts.build()
        pieces = []
        for _, group in forecasts.groupby('metric'):
            result = MissRiskModel(group, self.period.years).evaluate()
            pieces.append(result.loc[
                result.cutoff.le(LAST_SCORED_CUTOFF),
                ['metric', 'cik', 'quarter', 'date', 'cutoff', 'year',
                 'training', 'actual', 'consensus', 'predicted_correction',
                 'error_scale', 'scaled_gap', 'miss', 'model', 'history_rate']])
        scored = pd.concat(pieces, ignore_index=True)
        assert not scored.duplicated(['metric', 'cik', 'quarter']).any()
        assert scored.cutoff.between(self.period.start,
                                     LAST_SCORED_CUTOFF).all()
        assert scored.miss.eq(scored.actual.lt(scored.consensus)).all()
        assert scored.model.between(0, 1).all()
        scored.to_csv(predictions, index=False, mode='x')
        summaries, gaps, bands = [], [], []
        for metric, group in scored.groupby('metric'):
            for window, subset in [('full', group), ('without_2021',
                                   group.loc[group.year.ne(2021)])]:
                for period, rows in [('pooled', subset),
                                     *subset.groupby('year')]:
                    keys = dict(metric=metric, window=window, period=period)
                    gap, holds = self.gap_table(rows)
                    summaries.append(dict(**keys, **self.summarise(rows),
                                          gap_hypothesis=holds))
                    gaps.append(gap.assign(**keys))
                    bands.append(probability_bands(rows).assign(**keys))
        for suffix, table in [('summary', pd.DataFrame(summaries)),
                              ('gap_bands', pd.concat(gaps)),
                              ('probability_bands', pd.concat(bands))]:
            table.to_csv(self.output_dir / (self.PREFIX + suffix + '.csv'),
                         index=False, mode='x')
        return pd.DataFrame(summaries)
