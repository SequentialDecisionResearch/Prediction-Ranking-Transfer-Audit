# Prediction-Ranking-Transfer-Audit

Research code and experimental results accompanying the manuscript *When Prediction Transfer Does Not Imply Ranking Transfer: Explicit-Label-Free Learning Across Equity Identities*.

## Repository contents

- `transferlab/`: Python implementation of the evaluation procedures.
- `configs/`: Experiment configurations.
- `tests/`: Automated software tests.
- `results/h7/`, `results/h21/`, `results/h63/`: Frozen numerical results for the three forecast horizons.

## Installation and tests

Python 3.10 or later is recommended.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The experimental command-line interface is available through:

```bash
python -m transferlab run --help
```

## Data availability

The original market-data inputs are not included in this repository. Redistribution of third-party historical equity-price data may be subject to licensing restrictions.

The public repository contains the evaluation code, configurations, tests, and selected numerical output files. Re-running the complete market experiments requires compatible input data obtained separately.

## Interpretation

The analysis examines the relationship between return-prediction accuracy, cross-sectional ranking, and transfer across equity identities.

The reported V2 analyses are post-hoc sensitivity assessments. Calendar-block confidence intervals are exploratory because the evaluation covers a limited number of distinct calendar months.

The results do not establish a consistent ranking advantage or a deployable trading strategy.
