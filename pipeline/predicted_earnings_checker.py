import re
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .nlp import extract_eps_from_text

MAJOR_EXCHANGES = {'NMS','NYQ', 'NGM','NCM','ASE'}

def score_entities(entities_and_positions: list) -> str | None:
    scores = {}
    total=len(entities_and_positions)
    if total==0:
        return None
    for (org, label, position) in entities_and_positions:
        if label!='ORG':
            continue
        if org not in scores:
            scores[org] = 0
        
        position_weight = 1.0 - (0.9 * position/total)
        scores[org]+=1+position_weight
    sorted_orgs=sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_orgs

def extract_ticker_from_url(url: str) -> str | None:
    match = re.search(r'/[a-z0-9-]+-([A-Z]{1,5})-q\d+-\d{4}-earnings',url,re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def extract_ticker_from_text(text: str) -> str | None:
    pattern = re.compile(
        r'\((?:NYSE|NASDAQ|NYSEARCA)?:?\s*([A-Z]{1,5})\s*[+-]?[\d.]*%?\)',re.IGNORECASE)
    
    matches = pattern.findall(text)
    
    noise = {'ET', 'AI', 'IT', 'US', 'CEO', 'CFO', 'QA', 'OK'}
    
    for match in matches:
        ticker = match.upper().strip()
        if ticker not in noise and len(ticker) >= 2:
            return ticker
    
    return None
    
def get_analyst_estimate(ticker:str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.earnings_history
        if earnings is None:
            return {'success' : False, 'error' : 'No estimate data found'}
        
        estimated_eps = earnings.iloc[-1]

        return{'success' : True, 'estimated_eps': round(float(estimated_eps['epsEstimate']), 2), 'quarter': str(estimated_eps.name.date())}
    except Exception as e:
        return {'success': False, 'error':str(e)}
    
def run_earnings_analysis(ticker: str, doc) -> dict:
    estimate=get_analyst_estimate(ticker)
    if not estimate['success']:
        return{'success':False, 'error': estimate['error']}
    actual_eps=extract_eps_from_text(doc)

    if actual_eps is None:
        try:
            stock=yf.Ticker(ticker)
            earnings = stock.earnings_history
            recent=earnings.iloc[-1]
            actual_eps=round(float(recent['epsActual']),2)
            source = 'yfinance'
        except Exception:
            return {'success':False, 'error': 'Could not extract earnings'}
    else:
        source= 'NLP extraction'
    
    comparison = compare_eps(actual_eps , estimate['estimated_eps'])
    comparison['success']=True
    comparison['ticker']=ticker
    comparison['quarter']=estimate['quarter']
    comparison['source'] = source
    return comparison

    
def compare_eps(actual_eps:float, estimated_eps:float) -> dict:
    difference = round(actual_eps-estimated_eps, 2)
    surprise = round((difference/abs(estimated_eps))*100,2) if estimated_eps != 0 else None
    if actual_eps>estimated_eps:
        verdict='BEAT'
    elif actual_eps<estimated_eps:
        verdict='MISS'
    else:
        verdict='MET'
    return {'verdict': verdict, 'surprise': surprise, 'actual_eps':    actual_eps,
        'estimated_eps': estimated_eps,
        'difference':    difference}


def get_past_all_earnings(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.earnings_history
        if earnings is None or earnings.empty:
            return {'success' : False, 'error' : 'No estimate data found'}
        return {'success':True,'earnings':earnings}
    except Exception as e:
        return {'success': False, 'error':str(e)}
    

def plot_earnings_history(earnings):
    quarters=[str(d.date()) for d in earnings.index]
    estimates=earnings['epsEstimate'].tolist()
    actuals=earnings['epsActual'].tolist()
    beats = [a>=e for a,e in zip(actuals, estimates)]
    colours=['#1baf7a' if beat else '#e34948' for beat in beats]
    fx, ax = plt.subplots(figsize=(10,5))

    ax.plot(quarters, estimates, color = "#2a78d6", linestyle="--", marker='s', markersize=7,label='Analyst Estimate')

    for i, (q, actual, colour) in enumerate(zip(quarters, actuals, colours)):
        ax.scatter(q,actual, color=colour, s=120, zorder=5, edgecolors='white',linewidths=1.5 )

    ax.set_ylabel('EPS ($)' , fontsize=12)
    ax.set_xlabel('Quarter', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', color='#e1e0d9', linewidth=0.8)
    ax.set_facecolor('white')
    fx.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    beat_patch = mpatches.Patch(color='#1baf7a', label='Beat')
    miss_patch = mpatches.Patch(color='#e34948', label='Miss')
    est_line   = mpatches.Patch(color='#2a78d6', label='Analyst estimate')
    ax.legend(handles=[beat_patch, miss_patch, est_line], loc='upper left')
    plt.tight_layout()
    return fx