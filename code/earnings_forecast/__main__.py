"""Command line: reproduce the reported outputs and compare them.

Run from the ``code`` folder:

    python -m earnings_forecast reproduce --output-dir ../reproduced
    python -m earnings_forecast verify --output-dir ../reproduced
"""

import argparse
from pathlib import Path

from earnings_forecast.config import OUTPUT_DIR
from earnings_forecast.pipeline import Reproduction, compare_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=['reproduce', 'verify'])
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Empty folder for reproduced files')
    parser.add_argument('--reference-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.command == 'reproduce':
        Reproduction(args.output_dir).run()
    identical, different, missing = compare_outputs(args.output_dir,
                                                    args.reference_dir)
    print(f'{len(identical)} identical, {len(different)} different, '
          f'{len(missing)} without a reference file')
    for name in different:
        print(f'  different: {name}')
    for name in missing:
        print(f'  no reference: {name}')
    raise SystemExit(1 if different or missing else 0)


if __name__ == '__main__':
    main()
