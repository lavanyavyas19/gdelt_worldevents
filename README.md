# GDELT Event Intelligence Dashboard

A hybrid analytics dashboard + event intelligence system built with **Streamlit** and the **GDELT** dataset. Compares event patterns across **USA, India, and Iran** with burst detection, TF-IDF keyword extraction, and event-chain exploration.

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
│   └── processed/              # Generated parquet files
├── models/                     # TF-IDF vectorizer, burst rules
├── outputs/                    # Evaluation results
├── src/
│   ├── data_loader.py          # Load raw GDELT CSVs
│   ├── data_cleaning.py        # Clean, filter, type-convert
│   ├── feature_engineering.py  # QuadClass labels, actor pairs
│   ├── aggregation.py          # Monthly/weekly aggregation
│   ├── detect_bursts.py        # Z-score burst detection
│   ├── tfidf_module.py         # TF-IDF keyword extraction
│   ├── build_chain.py          # Event chain retrieval
│   ├── evaluation.py           # Retrieval-quality evaluation
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

## Data Source

[GDELT Project](https://www.gdeltproject.org/) — Global Database of Events, Language, and Tone.

Download daily event files from: https://data.gdeltproject.org/events/index.html
# gdelt_worldevents
