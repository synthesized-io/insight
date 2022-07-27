# insight

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/synthesized-io/insight/master.svg)](https://results.pre-commit.ci/latest/github/synthesized-io/insight/master)

## Installing the Python Package

```zsh
pip install insight
```

## Usage

### Metrics

At the core of insight are the metrics classes which can be evaluated on one series, two series,
one dataframe or two dataframes.

```pycon
>>> import insight.metrics as m
>>> metric = m.EarthMoversDistance()
>>> metric(df['A'], df['B'])
0.14
```

### Plotting

The package provides various plotting functions which allow you to easily explore any series, dataframe
or multiple dataframes.

```pycon
>>> import insight.plotting as p
>>> p.plot_dataset(df)
```
