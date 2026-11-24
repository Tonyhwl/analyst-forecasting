# Forecasting quarterly results from analyst consensus

Analysts' consensus estimates are biased in a systematic way: S&P 500 companies beat EPS consensus about 80% of the time. This project learns that bias. It forecasts quarterly **EPS, revenue and operating margin** by correcting consensus, and estimates the probability that a company **beats or misses**.

All results below are out of sample. Features and settings were selected using pre-2021 data only. Forecasts for 2021 to 19 November 2026 are walk-forward, and the miss-risk test was frozen and scored once. They use the S&P 500 roster fixed in January 2021 (including companies that later left the index).

## Results

| Result (2021–2026) | Value | 95% interval |
|---|---:|---:|
| EPS forecast error vs consensus | 16.57% lower | 14.21 to 18.93% |
| EPS forecast error vs consensus plus last four surprises | 5.69% lower | 3.85 to 7.54% |
| Revenue forecast error vs consensus | 13.98% lower | 12.28 to 15.59% |
| Revenue forecast error vs consensus plus last four surprises | 9.36% lower | 6.23 to 12.65% |
| Revenue misses correctly flagged in the riskiest 10% | 615 of 1,117 (55.1%), vs 42.3% using company history | +8.1 to +17.3 pp |
| Margin misses correctly flagged in the riskiest 10% | 771 of 1,091 (70.7%), vs 62.8% using company history | +4.8 to +10.9 pp |
| EPS beat probability AUC | 0.688, vs 0.667 using company history | |

EPS error was lowest in every test year. Predicted beat rates were within one percentage point of observed rates in all three beat bands. The miss-risk specification and pass rule were committed before the test was scored.

[Project report (PDF)](outputs/project-report.pdf)

## Method

| Output | Model | Inputs |
|---|---|---|
| EPS, revenue, margin forecast | Median gradient boosting predicting the correction to consensus | Past surprises, beat rate, seasonality, consensus level and year-on-year change |
| EPS beat probability | Logistic regression | EPS inputs plus the boosted forecast gap |
| Revenue and margin miss risk | Logistic regression | Company miss history plus the forecast gap |

- Data: S&P Capital IQ consensus (mean, one day before release) and reported normalized EPS, revenue and EBIT.
- Historical actuals come from a 2026 export and may include later restatements.
- Models refit every January on results announced before January 1. Every input uses only results released before the forecast date.
- Errors are compared on identical releases, with intervals clustered by announcement quarter.

## Code

`code/earnings_forecast` is the package behind every reported result.

| Module | Responsibility |
|---|---|
| `config` | Paths, evaluation periods, fixed model settings |
| `data` | S&P 500 rosters and Capital IQ release histories |
| `features` | Inputs built only from earlier releases |
| `models` | Boosted consensus corrections, beat and miss-risk models |
| `inference` | Paired comparisons with quarter-clustered intervals |
| `eps`, `explain`, `revenue_margin`, `miss_risk` | One study class per result |
| `pipeline` | Runs every study and compares with the saved outputs |

## Reproduce

The models use licensed S&P Capital IQ data, which this repository does not include. With a Capital IQ licence, export the two release files named in `code/earnings_forecast/config.py` into `data/`. The S&P 500 roster file is included.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd code
..\.venv\Scripts\python.exe -m unittest discover -s . -t .
..\.venv\Scripts\python.exe -m earnings_forecast reproduce --output-dir ..\reproduced
```

The unit tests run without the licensed data. `reproduce` rebuilds all 31 outputs and compares their values with `outputs/`. Release-level files are not published because they contain licensed values, so those are listed as having no reference file.

## Repository layout

| Folder | Contents |
|---|---|
| `code/earnings_forecast` | Package and unit tests |
| `outputs` | Report, summary tables and figures |
| `data` | S&P 500 rosters (January 2017 and 2021); licensed files go here |
