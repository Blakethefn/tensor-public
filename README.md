# tensor-public

`tensor-public` is the public, redacted, math-only companion to the private
Tensor project. It contains the theoretical manuscript and a small
reproducibility-oriented Python implementation of the mathematical identities
that the manuscript defines.

Sensitive information and operational code are intentionally omitted. This
repository contains no proprietary data, fitted forecasting system, backtest,
validated alpha strategy, security-specific forecast, trading signal, or
investment recommendation. It is not a production package and does not make
an empirical claim about any security or strategy.

## Scope

The manuscript source is [`docs/main.tex`](docs/main.tex), with shared LaTeX
definitions in [`docs/preamble.tex`](docs/preamble.tex). The package under
`src/tensor/` contains only public mathematical operations for event-scale
moment matching, explicit scenario probabilities, expected value and risk,
calibration metrics, convergence estimation, and uncertainty-aware sizing
identities.

All model-specific quantities must be supplied explicitly. In particular,
`h`, `rho`, `kappa`, scenario probabilities, neutral-state thresholds,
confidence thresholds, and valuation uncertainty have no public numerical
defaults. Numerical examples and test fixtures are clearly hypothetical rather
than empirical estimates.

The package preserves the distinctions between risk-neutral option moments and
physical probability estimates, and between valuation returns (`R^val`) and
realized price returns (`R^px`). The multinomial standard error is reported as
a sampling floor; fitted conditional probabilities require additional
coefficient, model, and dependence uncertainty.

## Explicit exclusions

The private project remains the authority for any private implementation. This
public repository excludes private data, databases, data-ingestion workflows,
forecast collection, deployed services, APIs, dashboards, bots, schedulers,
backups, credentials, deployment configuration, and security-sensitive
material. Those exclusions are deliberate and are not missing features.

## Reproduction

Install the package and test extra in a clean environment, then run:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

To rebuild the manuscript from the repository root:

```bash
mkdir -p docs/build
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=docs/build docs/main.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=docs/build docs/main.tex
```

The tracked document output is
`docs/build/earnings_valuation_model.pdf`. No Zenodo record is published by
this repository; [`CITATION.cff`](CITATION.cff) is prepared for a later DOI or
Zenodo connection.
