# GDELT Event Intelligence Dashboard

A Streamlit analytics dashboard for exploring geopolitical event patterns across **USA**, **India**, and **Iran** using structured event data from the [GDELT Project](https://www.gdeltproject.org/).

**Analysis window:** December 2025 – March 2026 (hard cutoff: 2026-03-26)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place GDELT daily CSV files in data/raw/
#    Download from: https://data.gdeltproject.org/events/index.html
#    Filenames follow the pattern: YYYYMMDD.export.CSV

# 3. Run the data pipeline (builds all processed files)
python -m src.prepare_data

# 4. Launch the dashboard
streamlit run app.py
```

> **First run note:** The pipeline will attempt to auto-download GDELT daily exports for Dec 2025–Mar 2026 if fewer than 90 raw CSV files are found. If downloads fail (network unavailable), place the CSV files manually in `data/raw/` before running.

---

## Project Structure

```
gdelt_dashboard/
├── app.py                      # Landing page / entry point
├── requirements.txt
├── data/
│   ├── raw/                    # GDELT daily .CSV exports (input)
│   └── processed/              # events.parquet + bursts.parquet (generated)
├── models/
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF model (generated)
│   └── burst_rules.json        # Saved burst detection parameters
├── outputs/
│   ├── cleaning_report.json    # Data quality stats from pipeline
│   └── cross_country_results.json  # Lead-lag correlation results
├── pages/
│   ├── 1_Overview.py           # Big-picture KPIs and country insights
│   ├── 2_Trends.py             # Weekly event volume and tone over time
│   ├── 3_Event_Types.py        # CAMEO event family breakdown
│   ├── 4_Map.py                # Geographic hotspot map
│   ├── 6_Burst_Dashboard.py    # Statistical spike detection
│   ├── 7_Event_Chain.py        # Causal chain explorer
│   └── 8_Topic_Lens.py         # TF-IDF keyword patterns
└── src/
    ├── config.py               # All constants, paths, thresholds
    ├── data_loader.py          # Load raw GDELT CSVs
    ├── preprocessing.py        # Clean, filter, feature engineering
    ├── aggregation.py          # Time-series roll-ups
    ├── burst.py                # Z-score burst detection
    ├── chains.py               # Multi-factor event chain scoring
    ├── chain_model.py          # Optional logistic regression chain scorer
    ├── keywords.py             # TF-IDF keyword extraction
    ├── cross_country.py        # Lead-lag correlation analysis
    ├── ingest.py               # GDELT download utilities
    ├── storage.py              # Parquet/pickle save+load with fallback
    ├── evaluation.py           # Pipeline quality checks
    ├── utils.py                # Shared Streamlit UI helpers
    └── prepare_data.py         # End-to-end pipeline runner
```

---

## Dashboard Pages

### Landing Page (`app.py`)
Entry point with feature cards linking to each view. Includes a recommended investigation workflow.

### 1 · Overview
**Question:** What's the big picture right now?

Displays headline KPIs (total events, average tone, spike days, conflict share) with first-half vs second-half delta comparisons. Below the metrics, per-country insight cards show activity change, dominant tone, and spike count. An auto-generated intelligence summary identifies the most notable cross-country patterns. Closes with a stacked bar chart showing conflict/cooperation composition by country.

### 2 · Trends
**Question:** How is activity changing over time?

Weekly event volume and average tone line charts for all selected countries. Spike weeks are annotated with red dashed markers. A toggle switches the volume chart to show cooperation share (0–1 ratio) instead of raw counts. Half-period trend arrows show whether each country's activity is rising or falling. An expandable section shows conflict vs cooperation composition as an area chart by week.

### 3 · Event Types
**Question:** What kinds of events are happening?

Breakdown of events across CAMEO event families — from verbal cooperation to material conflict. Filters by country and time period to see shifts in event type distribution.

### 4 · Event Map
**Question:** Where are events happening?

Geographic scatter map of event locations. Filter by country, event type, and time range to identify geographic hotspots. Capped at 15,000 points for rendering performance.

### 6 · Activity Spikes
**Question:** When did something unusual happen?

Z-score based statistical detection of days when event counts rose significantly above the rolling baseline. Three tabs:

- **Summary** — total spike days, strongest spike (σ), cross-country simultaneous spike calendar, event composition during spikes
- **Country Analysis** — per-country time series with confidence band (±2σ), rolling baseline, and spike markers
- **Spike Details** — sortable spike day table with severity labels, plus **Investigate →** buttons that jump directly to the Event Chain page pre-loaded to that date

Sidebar controls let you tune the detection threshold (σ) and baseline window (days) and recalculate on demand.

### 7 · Event Chain
**Question:** What happened before and after a specific event?

Given any anchor event, surfaces the most relevant preceding and following events scored across 16 factors: actor overlap, event family match, quad-class alignment, location proximity, tone distance, Goldstein scale distance, and temporal decay (τ = 3 days).

When navigated to via **Investigate →** from Activity Spikes, the page pre-filters to events within ±7 days of the spike date and displays a context banner. Results are displayed in three sections — **Previous Events**, **Anchor Event**, **Next Events** — with color-coded relevance cards showing score strength and factor breakdown.

### 8 · Topic Lens
**Question:** What metadata patterns dominate, and how do they shift during spikes?

TF-IDF distinctiveness scores across structured GDELT metadata (actor names, locations, event codes) — one tab per country. Each country tab shows:

- **Top Keywords Overall** — the most distinctive terms in the full dataset
- **Top Keywords During Spikes** — a diverging bar chart showing which terms rise or fall during spike days vs normal periods, with an auto-generated insight sentence

> Note: GDELT v1 does not include article text. These are metadata patterns, not topic models from news content.

---

## Recommended Workflow

1. **Overview** — Get the big picture: which country has the most activity, what's the overall tone, how many spikes occurred.
2. **Activity Spikes** → **Spike Details tab** — Find the most statistically unusual days.
3. Click **Investigate →** on any spike — jump to **Event Chain** pre-loaded to that date and country.
4. **Topic Lens** — Understand which actor and location patterns dominate during high-activity periods for each country.
5. **Trends** — Check whether activity is increasing or decreasing overall, and where tone is shifting.

---

## Data Pipeline

Running `python -m src.prepare_data` executes these steps in order:

| Step | Description | Output |
|------|-------------|--------|
| 0 | Download GDELT daily exports (skips if ≥90 CSVs found) | `data/raw/*.CSV` |
| 1 | Load and filter raw files to USA / India / Iran | in-memory DataFrame |
| 2 | Preprocess: parse dates, engineer features, apply cutoff | `outputs/cleaning_report.json` |
| 3 | Save processed events | `data/processed/events.parquet` |
| 4 | Burst detection (rolling mean + z-score) | `data/processed/bursts.parquet` |
| 5 | Fit TF-IDF vectorizer (800 features, unigram+bigram) | `models/tfidf_vectorizer.pkl` |
| 6 | Cross-country lead-lag correlation | `outputs/cross_country_results.json` |
| 7 | Chain model training (skipped — uses heuristic scoring) | — |
| 8 | Evaluation / quality checks | `outputs/evaluation_results.json` |
| 9 | — | — |

**Storage fallback:** The pipeline tries to save as Parquet (via `pyarrow`). If pyarrow is unavailable, it falls back to pickle automatically.

---

## Configuration (`src/config.py`)

All tunable parameters are in one place:

| Constant | Default | Description |
|----------|---------|-------------|
| `DATA_CUTOFF_DATE` | `"2026-03-26"` | Hard ceiling — no events beyond this date appear anywhere |
| `INGEST_START_DATE` | `"2025-12-01"` | Start of download range |
| `INGEST_END_DATE` | `"2026-03-31"` | End of download range |
| `COUNTRY_CODE_MAP` | `{"US","IN","IR"}` | GDELT country codes → display names |
| `BURST_ROLLING_WINDOW` | `7` | Days used to compute the rolling baseline |
| `BURST_Z_THRESHOLD` | `2.0` | Minimum z-score to flag a day as a spike |
| `BURST_MIN_EVENTS` | `5` | Minimum events required before a day can be flagged |
| `TFIDF_MAX_FEATURES` | `800` | Maximum vocabulary size for the TF-IDF model |
| `TFIDF_NGRAM_RANGE` | `(1, 2)` | Unigram + bigram features |
| `MAX_MAP_POINTS` | `15,000` | Cap on scatter map points for performance |

Burst detection thresholds can also be adjusted interactively in the Activity Spikes sidebar without re-running the pipeline.

---

## Source Data

**GDELT Project v1** — Global Database of Events, Language, and Tone.

Daily export files are tab-separated CSVs with 58 columns following the [GDELT 1.0 Event Codebook](http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf). Events are filtered to rows where `ActionGeo_CountryCode` is `US`, `IN`, or `IR`.

Download index: https://data.gdeltproject.org/events/index.html

---

## Dependencies

Key packages (see `requirements.txt` for full list):-

- `streamlit` ≥ 1.31 (required for `st.switch_page` and `st.page_link`)
- `pandas`, `numpy`
- `plotly`
- `scikit-learn` (TF-IDF vectorizer)
- `pyarrow` (optional — Parquet storage; falls back to pickle if missing)
- `pydeck` (map page)
