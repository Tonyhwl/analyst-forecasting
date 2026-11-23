"""Load the S&P 500 rosters and the Capital IQ release histories."""

import pandas as pd

from earnings_forecast.config import (
    CONSENSUS, DATA_DIR, EPS_FILE, REVENUE_EBIT_FILE, ROSTER_FILE,
)


class Roster:
    """The 500 SEC registrants in the S&P 500 at the start of a cohort year."""

    def __init__(self, cohort, data_dir=DATA_DIR):
        members = pd.read_csv(data_dir / ROSTER_FILE, dtype={'cik': str})
        self.members = members.loc[members.cohort.eq(cohort)]
        assert self.members.cik.nunique() == 500

    @property
    def ciks(self):
        return self.members.cik

    def tickers(self, separator='/'):
        return self.members.groupby('cik').ticker.agg(separator.join)

    def coverage(self, releases_by_cik):
        """List every roster issuer with its number of scored releases."""
        table = self.members.groupby('cik', as_index=False).agg(
            roster_tickers=('ticker', ', '.join),
            roster_name=('company', 'first'),
        )
        table['releases_scored'] = (
            table.cik.map(releases_by_cik).fillna(0).astype(int)
        )
        return table


class ReleaseData:
    """Quarterly consensus and actuals for one roster."""

    def __init__(self, roster, data_dir=DATA_DIR):
        self.roster = roster
        self.data_dir = data_dir

    def eps(self, end):
        """Complete normalised EPS pairs announced before ``end``."""
        raw = pd.read_csv(self.data_dir / EPS_FILE, dtype={'cik': str},
                          parse_dates=['announcement_date'])
        rows = raw.loc[
            raw.cik.isin(self.roster.ciks) & raw.status.eq('Complete')
            & raw.announcement_date.lt(end)
        ].copy()
        assert not rows.duplicated(['cik', 'fiscal_quarter']).any()
        assert rows[CONSENSUS].notna().all()
        return rows

    def revenue_and_margin(self, last_year):
        """Revenue and implied EBIT margin pairs, one row per metric."""
        raw = pd.read_csv(self.data_dir / REVENUE_EBIT_FILE,
                          dtype={'cik': str}, parse_dates=['announcement_date'])
        eligible = raw.loc[
            raw.cik.isin(self.roster.ciks)
            & raw.announcement_date.lt(pd.Timestamp(last_year + 1, 1, 1))
            & raw.revenue_actual_usd_m.gt(0)
            & raw.revenue_consensus_1d_usd_m.gt(0)
        ].copy()
        assert not eligible.duplicated(['cik', 'fiscal_quarter']).any()
        keys = eligible[['cik', 'fiscal_quarter', 'announcement_date']].rename(
            columns={'fiscal_quarter': 'quarter', 'announcement_date': 'date'})
        revenue = keys.assign(
            metric='Revenue', actual=eligible.revenue_actual_usd_m,
            consensus=eligible.revenue_consensus_1d_usd_m)
        margin = keys.assign(
            metric='Implied EBIT margin',
            actual=eligible.implied_ebit_margin_actual_pct,
            consensus=eligible.implied_ebit_margin_consensus_1d_pct)
        usable = eligible.margin_pair & ~eligible.zero_ebit_pair
        return pd.concat([revenue, margin.loc[usable]])
