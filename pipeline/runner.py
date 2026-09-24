import subprocess
import json
import sys,os


def run_pipeline_subprocess(url: str) -> dict:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = f"""
import sys
import json
sys.path.insert(0, {json.dumps(project_root)})

from pipeline.fetcher import fetch_page
from pipeline.extracter import extract_text
from pipeline.nlp import process_text
from pipeline.predicted_earnings_checker import extract_ticker_from_text, run_earnings_analysis, get_past_all_earnings,extract_ticker_from_url
from pipeline.topics import extract_topics

url = {json.dumps(url)}

html, error = fetch_page(url)
if error:
    print(json.dumps({{'success': False, 'error': error}}))
    sys.exit()

text, error = extract_text(html)
if error:
    print(json.dumps({{'success': False, 'error': error}}))
    sys.exit()
full_text = text

nlp_results = process_text(text)
ticker = extract_ticker_from_url(url)
if not ticker:
    ticker = extract_ticker_from_text(text)
if not ticker:
    print(json.dumps({{'success': False, 'needs_ticker': True, 'raw_text': text[:5000]}}))
    sys.exit()
topics = extract_topics(nlp_results['tokens'], company_name=ticker.lower())

earnings = run_earnings_analysis(ticker, nlp_results['doc'])

past = get_past_all_earnings(ticker)
if past['success']:
    df = past['earnings']
    past_earnings = {{'quarters':[str(d.date()) for d in df.index],'estimates': df['epsEstimate'].tolist(),'actuals': df['epsActual'].tolist()}}
else:
    past_earnings = None

output = {{'success':True,'ticker':ticker,'earnings': earnings,'past_earnings': past_earnings,'sentences': nlp_results['sentences'][:400],'word_count': nlp_results['word_count'],'sentence_count':nlp_results['sentence_count'],'key_sentences': nlp_results['key_sentences'],'entities':[(e[0], e[1]) for e in nlp_results['entities']],'raw_text':nlp_results['raw_text'],'topics':topics}}

print(json.dumps(output, separators=(',', ':')))
"""

    result = subprocess.run([sys.executable, '-c', script],capture_output=True,text=True,timeout=120)

    try:
        lines = result.stdout.strip().split('\n')
        json_line = next(line for line in reversed(lines) if line.startswith('{'))
        return json.loads(json_line)
    except (json.JSONDecodeError, StopIteration):
        return {'success': False,'error': f'Could not parse output: {result.stdout[:200]}','stderr': result.stderr[:500],'returncode': result.returncode}
    


