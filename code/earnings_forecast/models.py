"""Point forecasts and probability models, refitted each January."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from earnings_forecast.config import BOOSTING_SETTINGS
from earnings_forecast.features import SurpriseHistory


def january(year):
    return pd.Timestamp(year, 1, 1)


class ConsensusCorrection:
    """Median boosting of the scaled surprise added back to consensus."""

    def __init__(self):
        self.model = HistGradientBoostingRegressor(**BOOSTING_SETTINGS)

    def fit_predict(self, train, rows, actual, anchor, features):
        target = (train[actual] - train[anchor]) / train.scale
        self.model.fit(train[features], target)
        return rows[anchor] + rows.scale * self.model.predict(rows[features])


class AnnualEpsForecast:
    """Refit the EPS correction each January on releases already announced."""

    def __init__(self, actual, anchor, features):
        self.actual = actual
        self.anchor = anchor
        self.features = features

    def _fit_year(self, panel, year, block):
        train = panel.loc[panel.announcement_date.lt(january(year))]
        assert train.announcement_date.max() < block.as_of.min()
        return ConsensusCorrection().fit_predict(
            train, block, self.actual, self.anchor, self.features)

    def predict(self, panel, start):
        """Forecasts for every row dated on or after ``start``."""
        scored = panel.loc[panel.as_of.ge(start)]
        return pd.concat([self._fit_year(panel, year, block)
                          for year, block in scored.groupby(scored.as_of.dt.year)])

    def predict_all(self, panel, column):
        """Forecasts for every year with earlier training releases."""
        pieces = []
        for year, group in panel.groupby(panel.as_of.dt.year):
            train = panel.loc[panel.announcement_date.lt(january(year))]
            if train.empty:
                continue
            group = group.copy()
            group[column] = ConsensusCorrection().fit_predict(
                train, group, self.actual, self.anchor, self.features)
            pieces.append(group)
        if not pieces:
            raise ValueError('No prior years available for training')
        return pd.concat(pieces, ignore_index=True)


class BeatModel:
    """Logistic probability that EPS strictly beats consensus."""

    def __init__(self, features, first_year=2017):
        self.features = features
        self.first_year = first_year
        self.model = make_pipeline(
            SimpleImputer(strategy='median'), StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000),
        )

    def walk_forward(self, rows):
        probabilities = pd.Series(np.nan, index=rows.index)
        for year, group in rows.groupby(rows.as_of.dt.year):
            if year < self.first_year:
                continue
            train = rows.loc[rows.announcement_date.lt(january(year))]
            if train.beat.nunique() != 2:
                raise ValueError('Both beat outcomes are needed for training')
            self.model.fit(train[self.features], train.beat)
            probabilities.loc[group.index] = self.model.predict_proba(
                group[self.features])[:, 1]
        return probabilities

    @staticmethod
    def category(probability):
        """Label the more likely outcome from its beat probability."""
        if not probability.dropna().between(0, 1).all():
            raise ValueError('Beat probability must be between 0 and 1')
        labels = pd.Series('Coin flip', probability.index, dtype='string')
        for lower, upper, strength in [
            (.45, .55, 'Leaning'), (.25, .75, 'Confident'),
            (.10, .90, 'Very confident'),
        ]:
            labels.loc[probability.le(lower)] = strength + ' non-beat'
            labels.loc[probability.ge(upper)] = strength + ' beat'
        return labels.mask(probability.isna())

    @staticmethod
    def add_forecast_gap(rows, forecast, consensus):
        """Scale the forecast uplift by eight settled forecast errors."""
        parts = []
        for _, group in rows.groupby('cik'):
            group = group.sort_values('announcement_date')
            errors = (group.actual - group[forecast]).abs()
            history = group[['announcement_date']].copy()
            history['typical_error'] = errors.rolling(8).median().clip(
                lower=0.01)
            # Only errors settled before the forecast date are known.
            matched = pd.merge_asof(
                group.sort_values('as_of'), history, left_on='as_of',
                right_on='announcement_date', allow_exact_matches=False,
                suffixes=('', '_previous'),
            )
            matched['forecast_gap'] = (
                matched[forecast] - matched[consensus]
            ) / matched.typical_error
            parts.append(matched)
        return pd.concat(parts, ignore_index=True)


class SurpriseForecast:
    """Revenue or margin surprise from earlier surprises, refitted yearly."""

    @staticmethod
    def evaluate(observations, years):
        records = []
        for _, group in observations.groupby(['metric', 'cik']):
            records.extend(SurpriseHistory.features(group))
        features = pd.DataFrame(records)
        scores = []
        for _, group in features.groupby('metric'):
            for year in years:
                train = group[group.date < january(year)]
                test = group[group.date.dt.year.eq(year)].copy()
                assert train.date.max() < test.date.min()
                model = HistGradientBoostingRegressor(**BOOSTING_SETTINGS)
                model.fit(train[SurpriseHistory.COLUMNS], train.target)
                test['predicted_correction'] = model.predict(
                    test[SurpriseHistory.COLUMNS])
                scores.append(test)
        scored = pd.concat(scores)
        scored['consensus_error'] = scored.target.abs()
        scored['recent4_error'] = (scored.target - scored.recent4).abs()
        scored['model_error'] = (
            scored.target - scored.predicted_correction).abs()
        return scored


class MissRiskModel:
    """Probability of a revenue or margin miss from history and forecast gap."""

    def __init__(self, forecasts, years):
        self.years = years
        # Some issuers announce two fiscal quarters on one day. Both settle
        # together.
        self.rows = forecasts.assign(
            fiscal_order=forecasts.quarter.str[-4:] + forecasts.quarter.str[:3])
        self.rows = self.rows.sort_values(['date', 'fiscal_order'])
        assert self.rows.date.dt.year.between(2014, max(years)).all()
        assert not self.rows.duplicated(['cik', 'quarter']).any()

    def features(self):
        pieces = []
        for _, issuer in self.rows.groupby('cik'):
            history = pd.DataFrame({
                'released': issuer.date,
                'error_scale': issuer.model_error.rolling(8).median(),
            })
            rows = pd.merge_asof(issuer, history, left_on='cutoff',
                                 right_on='released', allow_exact_matches=False)
            pieces.append(rows.loc[rows.error_scale.gt(0)])
        rows = pd.concat(pieces, ignore_index=True)
        assert rows.released.lt(rows.cutoff).all()
        rows['scaled_gap'] = rows.predicted_correction / rows.error_scale
        # Preserve the gap's sign while limiting extreme ratios.
        rows['gap_input'] = np.arcsinh(rows.scaled_gap)
        return rows

    def evaluate(self):
        rows = self.features()
        folds = []
        for year in self.years:
            train = rows.loc[rows.date.lt(january(year))]
            test = rows.loc[rows.cutoff.dt.year.eq(year)].copy()
            assert train.date.max() < test.cutoff.min()
            assert train.miss.nunique() == 2
            test['year'] = year
            test['training'] = len(train)
            for name, features in [('rate_only', ['history_rate']),
                                   ('model', ['history_rate', 'gap_input'])]:
                fit = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
                fit.fit(train[features], train.miss)
                test[name] = fit.predict_proba(test[features])[:, 1]
            folds.append(test)
        return pd.concat(folds, ignore_index=True)
