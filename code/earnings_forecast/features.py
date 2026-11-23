"""Forecast inputs built only from results released before each forecast."""

import numpy as np
import pandas as pd

from earnings_forecast.config import ACTUAL, CONSENSUS


class CompanyHistory:
    """EPS inputs for one company, using releases before the forecast date."""

    def __init__(self, rows, actual=ACTUAL, consensus=CONSENSUS):
        self.actual = actual
        self.consensus = consensus
        self.rows = rows.sort_values('announcement_date').dropna(
            subset=[consensus, actual]
        ).copy()
        self.rows['surprise'] = self.rows[actual] - self.rows[consensus]

    def features(self, days_before=1):
        records = []
        for _, row in self.rows.iterrows():
            as_of = row.announcement_date - pd.Timedelta(days=days_before)
            past = self.rows.loc[self.rows.announcement_date.lt(as_of)]
            if len(past) >= 4:
                records.append(self.quarter(row, past, as_of))
        return pd.DataFrame(records)

    def quarter(self, row, past, as_of):
        consensus = self.consensus
        scale = max(past[consensus].tail(8).abs().median(), 0.01)
        surprises = past.surprise
        matching = past.loc[past.fiscal_quarter.str[:3].eq(
            row.fiscal_quarter[:3])]
        seasonal = matching.surprise.tail(3).median()
        change = (row[consensus] - matching[self.actual].iloc[-1]
                  if len(matching) else np.nan)
        return {
            'cik': row.cik, 'fiscal_quarter': row.fiscal_quarter,
            'announcement_date': row.announcement_date, 'as_of': as_of,
            self.actual: row[self.actual], consensus: row[consensus],
            'scale': scale,
            'median_bias_eps': row[consensus] + surprises.median(),
            'recent4_bias_eps': row[consensus] + surprises.tail(4).median(),
            'recent12_bias_eps': row[consensus] + surprises.tail(12).median(),
            'consensus_level': row[consensus] / scale,
            'last_surprise': surprises.iloc[-1] / scale,
            'median_four': surprises.tail(4).median() / scale,
            'median_twelve': surprises.tail(12).median() / scale,
            'beat_rate': surprises.tail(12).gt(0).mean(),
            'historical_beat_rate': surprises.gt(0).mean(),
            'seasonal_surprise': seasonal / scale,
            'eps_change': change / scale,
        }


class EpsPanel:
    """EPS inputs and scoring scale for every company in a release table."""

    @staticmethod
    def build(releases):
        parts = []
        for _, group in releases.groupby('cik'):
            group = group.sort_values('announcement_date', kind='stable')
            features = CompanyHistory(group).features()
            if features.empty:
                continue
            # Score errors against the median absolute actual of up to eight
            # releases announced before the forecast date.
            actual = group[ACTUAL].to_numpy()
            prior = group.announcement_date.searchsorted(
                features.as_of, side='left')
            features['scale_for_score'] = [
                max(0.01, np.median(np.abs(actual[max(0, i - 8):i])))
                for i in prior
            ]
            parts.append(features)
        return pd.concat(parts, ignore_index=True)


class SurpriseHistory:
    """Revenue or margin inputs from twelve earlier releases."""

    COLUMNS = ['last', 'recent4', 'recent12', 'beat_rate',
               'same_quarter_error', 'consensus_change', 'recent_dispersion']

    @staticmethod
    def features(group):
        rows = []
        group = group.sort_values('date')
        assert group.quarter.is_unique
        revenue = group.metric.iloc[0] == 'Revenue'
        error = group.actual - group.consensus
        # Revenue surprises are a percentage of consensus; margin surprises
        # are in percentage points.
        group['error'] = 100 * error / group.consensus if revenue else error
        for row in group.itertuples():
            past = group[group.date < row.date - pd.Timedelta(days=1)]
            fiscal, year = row.quarter.split()
            season = past[past.quarter.eq(f'{fiscal} {int(year) - 1}')]
            if len(past) < 12 or season.empty:
                continue
            prior = season.iloc[0]
            recent = past.error.tail(12)
            diff = row.consensus - prior.actual
            rows.append(dict(
                cik=row.cik, quarter=row.quarter, date=row.date,
                metric=row.metric, target=row.error,
                last=recent.iloc[-1], recent4=recent.tail(4).median(),
                recent12=recent.median(), beat_rate=recent.gt(0).mean(),
                same_quarter_error=prior.error,
                consensus_change=(100 * diff / prior.actual if revenue
                                  else diff),
                recent_dispersion=(recent - recent.median()).abs().median(),
            ))
        return rows
