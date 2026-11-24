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


def same_values(left, right, tolerance=1e-9):
    """Compare two CSV tables, allowing rounding and a different row order."""
    a, b = pd.read_csv(left), pd.read_csv(right)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False

    def equal(x, y):
        try:
            pd.testing.assert_frame_equal(x, y, check_exact=False,
                                          rtol=tolerance, atol=tolerance)
        except AssertionError:
            return False
        return True

    if equal(a, b):
        return True
    # Sort on labels and counts only: rounding could reorder float columns.
    keys = [col for col in a if not pd.api.types.is_float_dtype(a[col])]
    if not keys or a.duplicated(keys).any() or b.duplicated(keys).any():
        return False
    return equal(a.sort_values(keys, ignore_index=True),
                 b.sort_values(keys, ignore_index=True))


def compare_outputs(new_dir, reference_dir=OUTPUT_DIR):
    """Group reproduced files by how they compare with the reference."""
    result = {'identical': [], 'equal': [], 'different': [],
              'unavailable': []}
    for path in sorted(new_dir.iterdir()):
        reference = reference_dir / path.name
        if not reference.exists():
            status = 'unavailable'
        elif filecmp.cmp(path, reference, shallow=False):
            status = 'identical'
        elif path.suffix == '.csv' and same_values(path, reference):
            status = 'equal'
        else:
            status = 'different'
        result[status].append(path.name)
    return result
