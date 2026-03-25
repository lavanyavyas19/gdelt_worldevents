# GDELT Event Intelligence Dashboard

A research-quality analytics dashboard that explores global event patterns across **USA**, **India**, and **Iran** using the GDELT dataset. Features burst detection, event chain exploration, and TF-IDF keyword intelligence.

**Analysis window:** December – March

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place GDELT CSV files in data/raw/
#    Download from: https://data.gdeltproject.org/events/index.html

# 3. Run the data pipeline
python -m src.prepare_data

# 4. Launch the dashboard
streamlit run app.py
```

## Project Structure

```
gdelt_dashboard/
├── app.py                      # Main Streamlit entry point
├── requirements.txt
├── data/
│   ├── raw/                    # Place GDELT CSV files here
│   └── processed/              # Generated pickle/parquet files
├── models/                     # TF-IDF vectorizer, burst rules
├── outputs/                    # Evaluation results, cleaning report
├── src/
│   ├── config.py               # Central configuration & constants
│   ├── data_loader.py          # Load raw GDELT CSVs
│   ├── preprocessing.py        # Clean, filter, engineer features
│   ├── aggregation.py          # Time-series aggregation
│   ├── burst.py                # Z-score burst detection
│   ├── chains.py               # Scoring-based event chain retrieval
│   ├── keywords.py             # TF-IDF keyword extraction
│   ├── evaluation.py           # Quality & sanity check metrics
│   ├── storage.py              # Parquet/pickle save/load helpers
│   ├── utils.py                # Shared Streamlit page utilities
│   └── prepare_data.py         # End-to-end pipeline runner
└── pages/
    ├── 1_Overview.py
    ├── 2_Trends.py
    ├── 3_Event_Types.py
    ├── 4_Map.py
    ├── 5_Dataset_Explorer.py
    ├── 6_Burst_Dashboard.py
    ├── 7_Event_Chain.py
    └── 8_Topic_Lens.py
```

## Key Features

- **Burst Detection**: Z-score based spike detection with complete daily calendar, configurable thresholds, and per-country analysis
- **Event Chain Explorer**: Multi-factor relevance scoring (country, actors, event type, tone, proximity) with score explanations
- **Topic Lens**: TF-IDF keyword intelligence with country-specific stopword removal and normal vs burst keyword comparison
- **Dec–Mar Window**: Smart date filtering that identifies the dominant year and restricts to the correct analysis period

## Data Source

[GDELT Project](https://www.gdeltproject.org/) — Global Database of Events, Language, and Tone.

Download daily event files from: https://data.gdeltproject.org/events/index.html
