"""Explain the EPS corrections with TreeSHAP and partial dependence."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from threadpoolctl import threadpool_limits

from earnings_forecast.config import ACTUAL, CONSENSUS, EPS_FEATURES
from earnings_forecast.eps import FORECAST, KEYS, EpsStudy
from earnings_forecast.models import ConsensusCorrection, january

LABELS = {
    'consensus_level': 'Current consensus',
    'last_surprise': 'Latest surprise',
    'median_four': 'Median surprise, last four',
    'median_twelve': 'Median surprise, last twelve',
    'beat_rate': 'Beat frequency, last twelve',
    'seasonal_surprise': 'Same-quarter median surprise',
    'eps_change': 'Consensus less prior-year EPS',
}


class PartialDependence:
    """Average the fitted correction as one surprise measure varies."""

    features = ('median_four', 'last_surprise', 'median_twelve')

    def __init__(self, scored):
        # Use the central 90% of each input for display only.
        quantiles = np.linspace(0.05, 0.95, 31)
        self.grid = {f: np.unique(scored[f].quantile(quantiles))
                     for f in self.features}
        self.rows = []

    def add_year(self, model, block):
        for feature, grid in self.grid.items():
            inputs = block[EPS_FEATURES].copy()
            for value in grid:
                inputs[feature] = value
                self.rows.append((feature, value, model.predict(inputs).sum(),
                                  len(inputs)))

    def summary(self, period):
        values = pd.DataFrame(self.rows, columns=[
            'feature', 'value', 'adjustment_sum', 'releases'
        ])
        totals = values.groupby(['feature', 'value'], as_index=False).sum()
        totals['mean_adjustment'] = totals.adjustment_sum / totals.releases
        totals['period'] = period
        return totals.drop(columns='adjustment_sum')

    @classmethod
    def plot(cls, summary, path):
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), sharey=True,
                                 layout='constrained')
        for ax, feature in zip(axes, cls.features):
            values = summary.loc[summary.feature.eq(feature)]
            for period, label, colour, style in [
                ('development', '2017–2020 development', '#616f7a', '--'),
                ('later', '2021–2026 test', '#006b70', '-'),
            ]:
                curve = values.loc[values.period.eq(period)]
                ax.plot(100 * curve.value, 100 * curve.mean_adjustment,
                        color=colour, linestyle=style, linewidth=2,
                        marker='.', markersize=3, label=label)
            ax.set_title(LABELS[feature], fontsize=10)
            ax.axvline(0, color='#bbbbbb', linewidth=0.8, zorder=0)
            ax.spines[['right', 'top']].set_visible(False)
            ax.grid(axis='y', alpha=0.2)
        axes[0].set_ylabel('Mean forecast correction\n(% of EPS scale)')
        axes[0].legend(frameon=False, fontsize=8, loc='upper left')
        fig.supxlabel('Historical surprise (% of EPS scale)', fontsize=10)
        fig.savefig(path, dpi=180)
        plt.close(fig)


class ForecastExplanation:
    """TreeSHAP contributions for each annual EPS model in one period."""

    def __init__(self, period, output_dir, data_dir):
        self.period = period
        self.eps = EpsStudy(period, output_dir, data_dir)

    def run(self):
        panel = self.eps.panel()
        scored = panel.loc[panel.as_of.ge(self.period.start)]
        dependence = PartialDependence(scored)
        parts = []
        with threadpool_limits(limits=1):
            for year, block in scored.groupby(scored.as_of.dt.year):
                train = panel.loc[panel.announcement_date.lt(january(year))]
                parts.append(self.explain_year(train, block, dependence))
        out = pd.concat(parts, ignore_index=True)
        out['period'] = self.period.name
        self.check_saved(out)
        return out, dependence.summary(self.period.name)

    @staticmethod
    def explain_year(train, block, dependence):
        correction = ConsensusCorrection()
        prediction = correction.fit_predict(train, block, ACTUAL, CONSENSUS,
                                            EPS_FEATURES)
        dependence.add_year(correction.model, block)
        explainer = shap.TreeExplainer(
            correction.model, feature_perturbation='tree_path_dependent')
        explanation = explainer(block[EPS_FEATURES])
        # The model predicts a surprise divided by its historical EPS scale.
        out = block[['cik', 'fiscal_quarter', 'announcement_date', 'as_of',
                     CONSENSUS, 'scale']].copy()
        out['base_adjustment_eps'] = explanation.base_values * block.scale
        for j, feature in enumerate(EPS_FEATURES):
            out[feature + '_shap_eps'] = (
                explanation.values[:, j] * block.scale
            )
        out['forecast_eps'] = prediction
        terms = [name + '_shap_eps' for name in EPS_FEATURES]
        reconstructed = out[CONSENSUS] + out.base_adjustment_eps + out[
            terms].sum(axis=1, skipna=False)
        np.testing.assert_allclose(reconstructed, prediction, atol=1e-8,
                                   rtol=1e-10)
        return out

    def check_saved(self, out):
        saved = pd.read_csv(self.eps.forecast_file, dtype={'cik': str},
                            parse_dates=['announcement_date'],
                            usecols=KEYS + [FORECAST])
        joined = out.merge(saved, on=KEYS, how='outer', validate='one_to_one',
                           indicator=True)
        if not joined['_merge'].eq('both').all():
            raise ValueError('Release keys differ from the saved evaluation')
        np.testing.assert_allclose(joined.forecast_eps, joined[FORECAST],
                                   atol=1e-8, rtol=1e-10)


class ExplanationStudy:
    """SHAP shares and partial dependence for both periods."""

    def __init__(self, periods, output_dir, data_dir):
        self.periods = periods
        self.output_dir = output_dir
        self.data_dir = data_dir

    @staticmethod
    def importance(rows):
        terms = rows[[name + '_shap_eps' for name in EPS_FEATURES]].abs()
        terms.columns = EPS_FEATURES
        scaled = terms.div(rows.scale, axis=0)
        means = scaled.groupby(rows.period).mean()
        shares = means.div(means.sum(axis=1), axis=0) * 100
        summary = means.stack().rename('mean_abs_shap_scaled').to_frame()
        summary['share_pct'] = shares.stack()
        summary.index.names = ['period', 'feature']
        return summary.reset_index()

    @staticmethod
    def plot_importance(summary, path):
        pivot = summary.pivot(index='feature', columns='period',
                              values='share_pct')
        pivot = pivot.sort_values('later')
        y = np.arange(len(pivot))
        fig, ax = plt.subplots(figsize=(8, 4.3), layout='constrained')
        for period, offset, colour, label in [
            ('development', -0.17, '#84939c', '2017–2020 development'),
            ('later', 0.17, '#006b70', '2021–2026 test'),
        ]:
            bars = ax.barh(y + offset, pivot[period], 0.31, color=colour,
                           label=label)
            ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=8)
        ax.set_yticks(y, [LABELS[i] for i in pivot.index])
        ax.set_xlim(0, max(5, pivot.max().max() * 1.18))
        ax.set_xlabel('Share of mean absolute contribution to the '
                      'scaled adjustment (%)')
        ax.set_title('Which inputs moved the earnings forecast?', loc='left')
        ax.legend(frameon=False, loc='lower right', fontsize=8)
        ax.spines[['right', 'top']].set_visible(False)
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def run(self):
        results = [ForecastExplanation(p, self.output_dir, self.data_dir).run()
                   for p in self.periods]
        rows = pd.concat([rows for rows, _ in results], ignore_index=True)
        curves = pd.concat([curve for _, curve in results], ignore_index=True)
        summary = self.importance(rows)
        out = self.output_dir
        curves.to_csv(out / 'january_partial_dependence.csv', index=False)
        PartialDependence.plot(curves, out / 'january_partial_dependence.png')
        rows.to_csv(out / 'january_shap_by_release.csv', index=False)
        summary.to_csv(out / 'january_shap_importance.csv', index=False)
        self.plot_importance(summary, out / 'january_shap_importance.png')
        return summary
