# Capstone Part 2 sample – FreshBasket store checkouts

Teaching sample only. Uses a made-up grocery store dataset so it does not clash with the traffic assignment.

## Layout

```
sample_capstone_part2/
  data/freshbasket_checkouts.csv
  data/cleaned_checkouts.csv
  data/features_checkouts.csv
  figures/
  pipeline.py
  feature_engineering.py
  visualizations.py
  app.py
  pipeline.log
  report.md
  requirements.txt
```

## Setup

```
pip install -r requirements.txt
```

## Run

```
python pipeline.py
python feature_engineering.py
python visualizations.py
python app.py summary
python app.py by-hour --hour 11
python app.py recommend --day weekend
```

## Logging

- `logging.getLogger(__name__)` in each file
- console + `pipeline.log`
- format: time, level, module, message
- DEBUG = thresholds / maps; INFO = load/save/shape; WARNING = drops/imputes; ERROR = failures

## Demand category

Quartiles of `checkouts`: Low / Medium / High / Peak
