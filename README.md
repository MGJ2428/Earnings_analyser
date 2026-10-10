# Earnings Call Analyser

An NLP pipeline that fetches earnings call transcripts, performs per-speaker sentiment analysis, extracts key financial data, and tracks whether management tone predicts post-earnings stock returns.

Built as a portfolio project targeting software engineering and finance internships.

## What it does

Paste any Motley Fool earnings call transcript URL and the app will:

- **Detect the ticker** from the URL and cross-reference with yfinance
- **Extract EPS** from the transcript text via regex, falling back to yfinance `epsActual`
- **Compare actual vs analyst estimate** to produce a BEAT / MISS / MET verdict
- **Run FinBERT sentiment analysis** per speaker — CEO, CFO, and analyst Q&A separately
- **Detect speakers automatically** across different Motley Fool transcript formats (bold, plain, and double-dash)
- **Extract key themes** via LDA topic modelling, labelled into 2–3 word phrases by Claude Haiku
- **Store results** to a CSV and calculate Pearson correlation between CEO sentiment and 3-day post-earnings return
- **Plot a scatter chart** coloured by BEAT/MISS verdict with the current transcript point highlighted

## Architecture

URL input
└─► validate_url()
└─► fetch_page() [requests + trafilatura]
└─► extract_text()
└─► subprocess ──────────► process_text() [spaCy NER, tokenisation]
│ ├────────► extract_ticker_from_url()
│ ├────────► run_earnings_analysis() [EPS extraction + yfinance]
│ ├────────► get_past_all_earnings()
│ └────────► extract_topics() [Gensim LDA + Claude Haiku API]
└─► analyse_transcript_sentiment() [FinBERT, cached in Streamlit process]
└─► get_post_earnings_return() [yfinance price history]
└─► store_results() [CSV append]
└─► results page


**Key engineering decision:** spaCy causes a segmentation fault when loaded inside the Streamlit process on macOS. The entire NLP pipeline (spaCy, LDA, EPS extraction) runs in an isolated subprocess via `run_pipeline_subprocess()`. FinBERT runs in the main Streamlit process using `@st.cache_resource`.


## Tech stack

| Component | Library |
| Frontend | Streamlit |
| NLP / NER | spaCy (`en_core_web_sm`) |
| Sentiment | FinBERT (`ProsusAI/finbert`) |
| Topic modelling | Gensim LDA |
| Topic labelling | Anthropic API (Claude Haiku) |
| EPS data | yfinance |
| Correlation | scipy (Pearson r) |
| Visualisation | Plotly |
| Text extraction | trafilatura |


## File structure

Earnings_analyser/
├── app.py ← Streamlit homepage + pipeline orchestration
├── pages/
│ └── results.py ← Results display page
├── pipeline/
│ ├── fetcher.py ← HTTP fetch
│ ├── extracter.py ← Text extraction
│ ├── nlp.py ← spaCy processing + EPS regex
│ ├── sentiment.py ← FinBERT per-speaker sentiment
│ ├── predicted_earnings_checker.py ← EPS vs analyst estimate
│ ├── topics.py ← LDA + Claude Haiku labelling
│ ├── correlation.py ← Price return fetching + CSV storage + Plotly chart
│ ├── runner.py ← Subprocess isolation
│ └── validation.py
├── data/
│ └── correlation_data.csv ← ticker, date, sentiment, d1/d3 return, verdict
└── run_batch.py ← Batch processing script for multiple URLs


## Running locally

```bash
git clone https://github.com/YOUR_USERNAME/earnings-analyser
cd earnings-analyser
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
streamlit run app.py
```

Set your Anthropic API key for topic labelling:
```bash
export ANTHROPIC_API_KEY=your_key_here
```

Supports **Motley Fool** transcript URLs only (`fool.com/earnings/call-transcripts/...`).


## Key findings

Across 80+ earnings call transcripts spanning 2023–2026:

- Overall Pearson r between CEO sentiment score and 3-day post-earnings return: **−0.76**
- The relationship differs by verdict — positive sentiment in miss quarters shows a stronger negative return effect, consistent with market over-pricing of optimistic management tone
- FinBERT scores neutral for highly technical CEOs (e.g. Jensen Huang) — a known limitation of models trained primarily on analyst reports rather than executive language

## Limitations

- Only supports Motley Fool transcripts
- EPS figures from yfinance can differ slightly from transcript values due to basic vs diluted share count methodology
- Correlation is based on ~169 transcripts — statistically indicative but not conclusive
- FinBERT was fine-tuned on financial news, not earnings call language specifically
