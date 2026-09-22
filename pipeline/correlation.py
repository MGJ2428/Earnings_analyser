import os
import csv
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import re
from adjustText import adjust_text

data_path= '/Users/mayankjagadish/Desktop/Earnings_analyser/data/correlation_data.csv'
field_names = ['ticker', 'date', 'sentiment_score', 'd1_return', 'd3_return','verdict']

def extract_date_from_url(url: str) -> str | None:
    match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


def calculate_enhanced_correlation(df):
    beats = df[df['verdict'] == 'BEAT']
    misses = df[df['verdict'] == 'MISS']
    
    results = {}
    if len(beats) >= 3:
        r, p = stats.pearsonr(beats['sentiment_score'], beats['d3_return'])
        results['beat_r'] = round(r, 2)
        results['beat_n'] = len(beats)
    if len(misses) >= 3:
        r, p = stats.pearsonr(misses['sentiment_score'], misses['d3_return'])
        results['miss_r'] = round(r, 2)
        results['miss_n'] = len(misses)
    
    return results

def get_post_earnings_return(ticker: str, call_date: str) -> dict:
    try:
        date = datetime.strptime(call_date, '%Y-%m-%d')
        end = date + timedelta(days=10)
        stock = yf.Ticker(ticker)
        history = stock.history(start=date.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))

        if len(history)<2:
            return {'success': False, 'error':'Insufficient price data'}
        price_day_0=history['Close'].iloc[0]
        d1_return=None
        d3_return=None
        if len(history)>=2:
            d1_return=float(round(((history['Close'].iloc[1]-price_day_0)/price_day_0)*100,2))
        if len(history)>=4:
            d3_return=float(round(((history['Close'].iloc[3]-price_day_0)/price_day_0)*100,2))
        return{'success':True, 'd1_return':d1_return,'d3_return':d3_return}

    except Exception as e:
        return {'success': False, 'error': str(e)}

def store_results(tcker:str, date:str, sentiment_score:float, d1_return: float, d3_return: float, verdict:str) -> None:
    data_path= '/Users/mayankjagadish/Desktop/Earnings_analyser/data/correlation_data.csv'
    file_exists = os.path.exists(data_path) and os.path.getsize(data_path) > 0
    if file_exists:
        current_data=pd.read_csv(data_path)
        duplicate=((current_data['ticker']==tcker) & (current_data['date']==date)).any()
        if duplicate:
            return
    with open(data_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'ticker':tcker,'date':date,'sentiment_score':sentiment_score,'d1_return':d1_return,'d3_return':d3_return, 'verdict':verdict})

def calculate_correlations() -> dict:
    if not os.path.exists(data_path):
        return {'success': False, 'error': 'No data yet', 'n': 0}
    if os.path.getsize(data_path) == 0:
        return {'success': False, 'error': 'No data yet', 'n': 0}
    try:
        df = pd.read_csv(data_path).dropna(subset=['sentiment_score', 'd3_return'])
    except pd.errors.EmptyDataError:
        return {'success': False, 'error': 'No data yet', 'n': 0}

    df = df.drop_duplicates(subset=['ticker', 'date'])

    if len(df) < 5:
        return {
            'success': False,
            'error':   f'Need at least 5 data points — have only {len(df)} so far',
            'n':       len(df)
        }
    r, p_value = stats.pearsonr(df['sentiment_score'], df['d3_return'])

    beat_r = beat_n = miss_r = miss_n = None

    if 'verdict' in df.columns:
        beats  = df[df['verdict'] == 'BEAT'].dropna(subset=['sentiment_score', 'd3_return'])
        misses = df[df['verdict'] == 'MISS'].dropna(subset=['sentiment_score', 'd3_return'])
        if len(beats) >= 3:
            beat_r, _ = stats.pearsonr(beats['sentiment_score'], beats['d3_return'])
            beat_r    = round(beat_r, 2)
            beat_n    = len(beats)
        if len(misses) >= 3:
            miss_r, _ = stats.pearsonr(misses['sentiment_score'], misses['d3_return'])
            miss_r    = round(miss_r, 2)
            miss_n    = len(misses)

    return {
        'success':     True,
        'r':           round(r, 2),
        'p_value':     round(p_value, 3),
        'n':           len(df),
        'significant': p_value < 0.05,
        'beat_r':      beat_r,
        'beat_n':      beat_n,
        'miss_r':      miss_r,
        'miss_n':      miss_n,
        'data':        df.to_dict('records')
    }

def plot_correlation(data: list, highlight_ticker: str = None, highlight_date: str = None) -> plt.Figure:
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    fig, ax = plt.subplots(figsize=(7, 4))
    texts = []

    for ignore, row in df.iterrows():
        is_highlight = (
            highlight_ticker and highlight_date and
            row['ticker'] == highlight_ticker and
            row['date'] == highlight_date
        )

        colour = '#1baf7a' if row.get('verdict') == 'BEAT' else '#e34948' if row.get('verdict') == 'MISS' else '#2A78D6'

        if is_highlight:
            ax.scatter(row['sentiment_score'], row['d3_return'],color=colour, s=400, alpha=0.15, edgecolors='none', zorder=4)
            ax.scatter(row['sentiment_score'], row['d3_return'],color=colour, s=200, alpha=0.25, edgecolors='none', zorder=4)
            ax.scatter(row['sentiment_score'], row['d3_return'],color=colour, s=120, alpha=1.0,edgecolors='white', linewidths=2.5, zorder=5)
            ax.annotate(f"{row['ticker']} < current",(row['sentiment_score'], row['d3_return']),textcoords="offset points",xytext=(10, 5), fontsize=8,color=colour, fontweight='bold')
        else:
            ax.scatter(row['sentiment_score'], row['d3_return'],color=colour, s=90, alpha=0.7,edgecolors='white', linewidths=1.5, zorder=3)
            ax.annotate(row['ticker'],(row['sentiment_score'], row['d3_return']),textcoords="offset points",xytext=(6, 4), fontsize=8, color='#6B7A99')
    adjust_text(texts,arrowprops=dict(arrowstyle='-', color='#3A4A66', lw=0.5))
    if len(df) >= 3:
        z = np.polyfit(df['sentiment_score'], df['d3_return'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['sentiment_score'].min(),df['sentiment_score'].max(), 100)
        ax.plot(x_line, p(x_line), color='#6B7A99',linestyle='--', linewidth=1.5)

    ax.axhline(y=0, color='#3A4A66', linewidth=0.8)
    ax.axvline(x=0, color='#3A4A66', linewidth=0.8)
    ax.set_xlabel('FinBERT Sentiment Score', fontsize=11)
    ax.set_ylabel('3-Day Post-Earnings Return (%)', fontsize=11)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='#e1e0d9', linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    return fig