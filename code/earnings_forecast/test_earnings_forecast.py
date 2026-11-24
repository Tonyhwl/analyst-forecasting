"""Unit tests: no future information enters any input, and the statistics
behave as documented."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from earnings_forecast.config import ACTUAL, CONSENSUS, EPS_FEATURES
from earnings_forecast.features import CompanyHistory, EpsPanel, SurpriseHistory
from earnings_forecast.inference import (
    bootstrap_relative_reduction, relative_reduction,
)
from earnings_forecast.models import (
    AnnualEpsForecast, BeatModel, MissRiskModel,
)


def eps_releases(quarters=24, cik='0000000001'):
    periods = pd.period_range('2010Q1', periods=quarters, freq='Q')
    n = np.arange(quarters)
    return pd.DataFrame({
        'cik': cik,
        'fiscal_quarter': [f'FQ{p.quarter} {p.year}' for p in periods],
        'announcement_date': periods.end_time.normalize() + pd.Timedelta(days=30),
        CONSENSUS: 1 + n / 10,
        ACTUAL: 1 + n / 10 + (n % 5) / 20,
    })


class CompanyHistoryTests(unittest.TestCase):
    def test_release_on_forecast_date_is_excluded(self):
        raw = eps_releases(12)
        target = raw.announcement_date.iloc[-1]
        # Move the previous release to the day before the target release,
        # which is the forecast date itself.
        raw.loc[10, 'announcement_date'] = target - pd.Timedelta(days=1)
        row = CompanyHistory(raw).features().iloc[-1]
        expected = (raw[ACTUAL] - raw[CONSENSUS]).iloc[9]
        self.assertAlmostEqual(row.last_surprise * row.scale, expected)

    def test_future_outcomes_do_not_change_features(self):
        raw = eps_releases(16)
        before = CompanyHistory(raw.iloc[:12]).features()
        changed = raw.copy()
        changed.loc[12:, ACTUAL] += 100
        after = CompanyHistory(changed).features().iloc[:len(before)]
        pd.testing.assert_frame_equal(before[EPS_FEATURES],
                                      after[EPS_FEATURES])

    def test_score_scale_uses_only_earlier_actuals(self):
        raw = eps_releases(14)
        panel = EpsPanel.build(raw)
        last = panel.iloc[-1]
        earlier = raw.loc[raw.announcement_date.lt(last.as_of), ACTUAL]
        self.assertAlmostEqual(last.scale_for_score,
                               max(0.01, earlier.tail(8).abs().median()))


class AnnualFitTests(unittest.TestCase):
    def test_fit_excludes_releases_from_the_forecast_year(self):
        panel = EpsPanel.build(pd.concat(
            [eps_releases(40, cik=str(i)) for i in range(3)]))
        seen = []

        def capture(self, train, rows, actual, anchor, features):
            seen.append((train.announcement_date.max(), rows.as_of.min()))
            return rows[anchor]

        with patch('earnings_forecast.models.ConsensusCorrection.fit_predict',
                   capture):
            AnnualEpsForecast(ACTUAL, CONSENSUS, EPS_FEATURES).predict(
                panel, pd.Timestamp('2016-01-01'))
        self.assertTrue(seen)
        for latest_training, first_forecast in seen:
            self.assertLess(latest_training,
                            pd.Timestamp(first_forecast.year, 1, 1))


class GapTests(unittest.TestCase):
    def test_gap_ignores_errors_not_yet_settled(self):
        dates = pd.date_range('2015-01-31', periods=12, freq='QE')
        rows = pd.DataFrame({
            'cik': 'A', 'announcement_date': dates,
            'as_of': dates - pd.Timedelta(days=1),
            'actual': 1.0, CONSENSUS: 1.0, 'forecast': 1.1,
        })
        rows.loc[11, 'actual'] = 50.0  # the current release's own outcome
        gapped = BeatModel.add_forecast_gap(rows, 'forecast', CONSENSUS)
        self.assertAlmostEqual(gapped.typical_error.iloc[-1], 0.1)

    def test_miss_scale_excludes_the_current_release(self):
        dates = pd.date_range('2014-03-31', periods=12, freq='QE')
        forecasts = pd.DataFrame({
            'cik': 'A', 'quarter': [f'FQ{d.quarter} {d.year}' for d in dates],
            'date': dates, 'cutoff': dates - pd.Timedelta(days=1),
            'model_error': 1.0, 'predicted_correction': 0.5,
            'miss': 0, 'history_rate': 0.3,
        })
        forecasts.loc[11, 'model_error'] = 99.0
        rows = MissRiskModel(forecasts, range(2016, 2017)).features()
        self.assertAlmostEqual(rows.error_scale.iloc[-1], 1.0)


class SurpriseHistoryTests(unittest.TestCase):
    def test_inputs_use_releases_before_the_forecast_date(self):
        dates = pd.date_range('2010-03-31', periods=16, freq='QE')
        group = pd.DataFrame({
            'cik': 'A', 'metric': 'Revenue', 'date': dates,
            'quarter': [f'FQ{d.quarter} {d.year}' for d in dates],
            'consensus': 100.0, 'actual': 101.0,
        })
        group.loc[15, 'actual'] = 200.0
        rows = pd.DataFrame(SurpriseHistory.features(group))
        self.assertAlmostEqual(rows.iloc[-1]['last'], 1.0)
        self.assertAlmostEqual(rows.iloc[-1].target, 100.0)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.clusters = pd.Series(np.repeat(np.arange(8), 10))

    def test_relative_reduction_point_estimate(self):
        benchmark = pd.Series(np.linspace(1, 2, 80))
        gain, low, high = relative_reduction(benchmark, benchmark / 2,
                                             self.clusters)
        self.assertAlmostEqual(gain, 50.0)
        self.assertLessEqual(low, gain)
        self.assertGreaterEqual(high, gain)

    def test_equal_losses_give_zero(self):
        loss = pd.Series(np.ones(80))
        self.assertEqual(relative_reduction(loss, loss, self.clusters),
                         (0.0, 0.0, 0.0))

    def test_bootstrap_equal_totals(self):
        totals = np.ones((16, 2))
        np.testing.assert_allclose(bootstrap_relative_reduction(totals), 0)


class CategoryTests(unittest.TestCase):
    def test_bands(self):
        p = pd.Series([.95, .8, .6, .5, .4, .2, .05])
        self.assertEqual(BeatModel.category(p).tolist(), [
            'Very confident beat', 'Confident beat', 'Leaning beat',
            'Coin flip', 'Leaning non-beat', 'Confident non-beat',
            'Very confident non-beat'])


class CompareOutputTests(unittest.TestCase):
    def compare(self, new, reference):
        # Imported here because the pipeline also loads SHAP and matplotlib.
        from earnings_forecast.pipeline import same_values
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / name for name in ('new.csv', 'ref.csv')]
            for path, table in zip(paths, (new, reference)):
                table.to_csv(path, index=False)
            return same_values(*paths)

    def test_rounding_and_row_order_are_accepted(self):
        table = pd.DataFrame({'band': [0, 10], 'rate': [0.1, 0.2]})
        moved = table.iloc[::-1].assign(rate=lambda x: x.rate + 1e-13)
        self.assertTrue(self.compare(moved, table))

    def test_changed_value_is_rejected(self):
        table = pd.DataFrame({'band': [0, 10], 'rate': [0.1, 0.2]})
        self.assertFalse(self.compare(table.assign(rate=[0.1, 0.2001]), table))


if __name__ == '__main__':
    unittest.main()
