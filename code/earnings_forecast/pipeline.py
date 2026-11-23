"""Run every study behind the reported results, in dependency order."""

import filecmp

import pandas as pd

from earnings_forecast.config import (
    DATA_DIR, LATER, OUTPUT_DIR, PERIODS,
)
from earnings_forecast.eps import BeatStudy, EpsStudy
from earnings_forecast.explain import ExplanationStudy
from earnings_forecast.miss_risk import (
    DevelopmentCalibration, DevelopmentMissRisk, FrozenMissRiskTest,
)
from earnings_forecast.revenue_margin import CompanyBreadth, RevenueMarginStudy


class Reproduction:
    """Recreate the reported outputs in an empty directory."""

    def __init__(self, output_dir, data_dir=DATA_DIR):
        self.output_dir = output_dir
        self.data_dir = data_dir

    def steps(self):
        out, data = self.output_dir, self.data_dir
        for period in PERIODS:
            yield f'EPS forecasts, {period.name}', EpsStudy(period, out, data).run
        yield 'SHAP explanations', ExplanationStudy(PERIODS, out, data).run
        yield 'Beat probabilities', self._beat
        for period in PERIODS:
            yield (f'Revenue and margin, {period.name}',
                   RevenueMarginStudy(period, out, data).run)
        yield 'Company breadth, later', CompanyBreadth(LATER, out, data).run
        yield 'Miss risk, development', DevelopmentMissRisk(out, data).run
        yield 'Miss-risk calibration', DevelopmentCalibration(out, data).run
        yield 'Frozen miss-risk test', FrozenMissRiskTest(LATER, out, data).run

    def _beat(self):
        seven, comparisons = [], []
        for period in PERIODS:
            summary, gap = BeatStudy(period, self.output_dir,
                                     self.data_dir).run()
            seven.append(summary)
            comparisons.extend(gap)
        pd.DataFrame(seven).to_csv(
            self.output_dir / 'january_beat_summary.csv', index=False)
        pd.DataFrame(comparisons).to_csv(
            self.output_dir / 'january_beat_gap_summary.csv', index=False)

    def run(self, log=print):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(self.output_dir.iterdir()):
            raise FileExistsError(f'{self.output_dir} is not empty')
        for name, step in self.steps():
            log(f'Running: {name}')
            step()
        return sorted(p.name for p in self.output_dir.iterdir())


def compare_outputs(new_dir, reference_dir=OUTPUT_DIR):
    """Return (identical, different, missing) file names."""
    identical, different, missing = [], [], []
    for path in sorted(new_dir.iterdir()):
        reference = reference_dir / path.name
        if not reference.exists():
            missing.append(path.name)
        elif filecmp.cmp(path, reference, shallow=False):
            identical.append(path.name)
        else:
            different.append(path.name)
    return identical, different, missing
