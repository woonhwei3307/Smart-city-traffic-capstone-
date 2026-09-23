# Capstone Part 2 – Python Building a Reproducible Traffic Analysis Pipeline


## Layout

```
sample_capstone_part2/
  data/Metro_Interstate_Traffic_Volume.csv
  data/Cleaned_metro_interstate_traffic_volume.csv
  data/features_metro_interstate_traffic_volume.csv
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
python app.py query-time --datetime "2012-10-02 09:00:00"
python app.py high-traffic --congestion Severe
python app.py compare-days
```

## Logging

- `logging.getLogger(__name__)` in each file
- console + `pipeline.log`
- format: time, level, module, message
- DEBUG = thresholds / maps; INFO = load/save/shape; WARNING = drops/imputes; ERROR = failures

## Congestion category

Quartiles of `checkouts`: Low / Medium / High / Severe
