import streamlit as st
import os
from pipeline.runner import run_pipeline_subprocess
from pipeline.validation import validate_url
from pipeline.sentiment import analyse_transcript_sentiment
from pipeline.correlation import extract_date_from_url, get_post_earnings_return, store_results, calculate_correlations
st.set_page_config(page_title="Earnings Analyser", layout='wide')

@st.cache_resource(show_spinner="Loading FinBERT... (first time only, ~2 minutes)")
def load_finbert():
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ['HF_HOME'] = os.path.join(base_dir, '.hf_cache')

finbert = load_finbert()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #E8EAF0; }
[data-testid="stVerticalBlockBorderWrapper"] { background: #0E1628; border: 0.5px solid #1E2A45 !important; border-radius: 12px !important; }
.stTextInput input { background: #0E1628; border-color: #1E2A45; color: #E8EAF0; }
.stTabs [data-baseweb="tab"] { color: #6B7A99; }
</style>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.5, 1], gap='large')

with col_left:
    st.markdown('NLP powered analysis')
    st.title("Decode Earnings Calls with AI")

    tab_url, = st.tabs(["Website URL"])

    with tab_url:
        url = st.text_input('Report URL', placeholder='https://fool.com/earnings/...')

        if st.button('Analyse URL', disabled=not url):
            with st.spinner("Validating URL..."):
                valid, msg = validate_url(url)

            if not valid:
                st.error(msg)
            else:
                with st.spinner("Fetching and analysing... (this may take 30 seconds)"):
                    result = run_pipeline_subprocess(url)
                if not result['success']:
                    st.error(result.get('error'))
                    st.code(result.get('stderr', ''))
                    st.write("returncode:", result.get('returncode'))
                if not result['success']:
                    if result.get('needs_ticker'):
                        st.session_state['needs_ticker'] = True
                        st.session_state['raw_text'] = result.get('raw_text', '')
                        st.session_state['source_url'] = url
                    else:
                        st.error(result.get('error', 'Something went wrong'))
                else:
                    with st.spinner("Running sentiment analysis..."):
                        sentiment = analyse_transcript_sentiment(
                            result.get('raw_text', ''),
                            result.get('sentences', []),
                            finbert
                        )
                    nlp_data = {
                        'sentiment':sentiment,
                        'word_count':result.get('word_count', 0),
                        'sentence_count':result.get('sentence_count', 0),
                        'key_sentences':result.get('key_sentences', []),
                        'entities':result.get('entities', []),
                        'past_earnings':result.get('past_earnings'),
                        'sentences':result.get('sentences', []),
                        'topics': result.get('topics', [])
                    }
                    with st.spinner('Calculating post_earnings return ...'):
                        call_date=extract_date_from_url(url)
                        st.session_state['call_date'] = call_date
                        if call_date and result['ticker']:
                            returns=get_post_earnings_return(result['ticker'], call_date)
                            print(f"DEBUG returns: {returns}")
                            if returns['success']:
                                print(f"DEBUG about to store: ticker={result['ticker']}, date={call_date}, verdict={result.get('earnings', {}).get('verdict', 'UNKNOWN')}")
                                store_results(tcker=result['ticker'],date=call_date, sentiment_score=sentiment.get('overall', {}).get('overall', 0.0), d1_return=returns['d1_return'], d3_return=returns['d3_return'],verdict=result.get('earnings', {}).get('verdict', 'UNKNOWN'))

                    st.session_state['ticker'] = result['ticker']
                    st.session_state['earnings'] = result['earnings']
                    st.session_state['nlp_results'] = nlp_data
                    st.session_state['source_url'] = url
                    st.session_state['raw_text'] = result.get('raw_text', '')
                    st.session_state['needs_ticker'] = False
                    st.switch_page('pages/results.py')

        if st.session_state.get('needs_ticker', False):
            st.warning("Could not auto-detect the company ticker.")
            col_1, col_2 = st.columns([2, 1])
            with col_1:
                manual_ticker = st.text_input(
                    'Enter ticker manually',
                    placeholder='e.g. AAPL, MSFT, ORCL',
                    key='manual_ticker'
                ).upper().strip()
            with col_2:
                st.write('')
                st.write('')
                if st.button('Confirm', disabled=not manual_ticker):
                    with st.spinner(f'Checking earnings for {manual_ticker}...'):
                        result = run_pipeline_subprocess(st.session_state['source_url'])
                    st.session_state['ticker'] = manual_ticker
                    st.session_state['earnings'] = result.get('earnings')
                    st.session_state['needs_ticker'] = False
                    st.switch_page('pages/results.py')
    st.markdown("""
    <p style="font-size:12px; font-weight:500; color:#7F77DD; margin:1.5rem 0 6px;">Recommended sources</p>
    <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:8px;">
        <a href="https://www.fool.com/earnings-call-transcripts/" target="_blank"
            style="background:#0E1628; border:0.5px solid #1E2A45; border-radius:10px; padding:12px 14px; text-decoration:none; display:block;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:13px; font-weight:500; color:#C8D0E0;">Motley Fool</span>
                <span style="font-size:10px; padding:2px 7px; border-radius:20px; background:#0B2A1E; color:#4CAF87; border:0.5px solid #1D5C40;">Works</span>
            </div>
            <div style="font-size:11px; color:#3A4A66;">Full transcripts, static HTML</div>
            <div style="font-size:11px; color:#2A78D6; margin-top:4px;">↗ fool.com</div>
        </a>
        <div style="background:#0E1628; border:0.5px solid #1E2A45; border-radius:10px; padding:12px 14px; opacity:0.3;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:13px; font-weight:500; color:#C8D0E0;">SEC EDGAR</span>
                <span style="font-size:10px; padding:2px 7px; border-radius:20px; background:#2A0E0E; color:#CF6679; border:0.5px solid #5C1D28;">Incompatible</span>
            </div>
            <div style="font-size:11px; color:#3A4A66;">Different format, avoid</div>
        </div>
        <div style="background:#0E1628; border:0.5px solid #1E2A45; border-radius:10px; padding:12px 14px; opacity:0.3;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:13px; font-weight:500; color:#C8D0E0;">Seeking Alpha</span>
                <span style="font-size:10px; padding:2px 7px; border-radius:20px; background:#2A0E0E; color:#CF6679; border:0.5px solid #5C1D28;">Blocked</span>
            </div>
            <div style="font-size:11px; color:#3A4A66;">Blocks scrapers, avoid</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_right:
    nlp = st.session_state.get('nlp_results', {})
    sentiment = nlp.get('sentiment')
    earnings  = st.session_state.get('earnings')

    with st.container(border=True):
        if sentiment:
            score = sentiment.get('ceo', {}).get('overall', 0.0)
            label = sentiment.get('ceo', {}).get('label', 'neutral').capitalize()
            st.metric("Sentiment (CEO)", f"{score:+.2f}", label)
        else:
            st.metric("Sentiment (CEO)", "+0.72", "Example")

    with st.container(border=True):
        if earnings and earnings.get('success'):
            st.metric("Earnings verdict",earnings.get('verdict', 'N/A'),f"{earnings.get('surprise', 0):+.1f}% surprise")
        else:
            st.metric("Earnings verdict", "N/A", "No data")

    with st.container(border=True):
        corr = calculate_correlations()
        if corr.get('success'):
            st.metric("Price correlation", f"r = {corr['r']:+.2f}",f"{corr['n']} transcripts")
        else:
            st.metric("Price correlation", "r = —", "Analyse more transcripts")
    topics = nlp.get('topics', [])
    if topics:
        topic_html = ''
        for t in topics[:4]:
            label = t.get('label', '')
            words = ' · '.join(t.get('words', [])[:3])
            topic_html += (
                f'<div style="background:#141D30; border:0.5px solid #1E2A45;'
                f'border-radius:8px; padding:8px 10px;">'
                f'<div style="font-size:10px; color:#3A4A66; margin-bottom:3px;">Topic {t["id"]}</div>'
                f'<div style="font-size:12px; font-weight:500; color:#C8D0E0;">{label}</div>'
                f'<div style="font-size:10px; color:#3A4A66; margin-top:3px;">{words}</div>'
                f'</div>'
            )
        badge = f"{len(topics)} found"
    else:
        topic_html = (
            '<div style="background:#141D30; border:0.5px solid #1E2A45; border-radius:8px; padding:8px 10px;">'
            '<div style="font-size:10px; color:#3A4A66; margin-bottom:3px;">Topic 1</div>'
            '<div style="font-size:12px; font-weight:500; color:#C8D0E0;">AI Strategy</div>'
            '</div>'
            '<div style="background:#141D30; border:0.5px solid #1E2A45; border-radius:8px; padding:8px 10px;">'
            '<div style="font-size:10px; color:#3A4A66; margin-bottom:3px;">Topic 2</div>'
            '<div style="font-size:12px; font-weight:500; color:#C8D0E0;">Revenue Growth</div>'
            '</div>'
            '<div style="background:#141D30; border:0.5px solid #1E2A45; border-radius:8px; padding:8px 10px;">'
            '<div style="font-size:10px; color:#3A4A66; margin-bottom:3px;">Topic 3</div>'
            '<div style="font-size:12px; font-weight:500; color:#C8D0E0;">Supply Chain</div>'
            '</div>'
            '<div style="background:#141D30; border:0.5px solid #1E2A45; border-radius:8px; padding:8px 10px;">'
            '<div style="font-size:10px; color:#3A4A66; margin-bottom:3px;">Topic 4</div>'
            '<div style="font-size:12px; font-weight:500; color:#C8D0E0;">Market Expansion</div>'
            '</div>'
        )
        badge = "example"

    st.markdown(
        f'<div style="background:#0E1628; border:0.5px solid #1E2A45; border-radius:12px; padding:14px 16px; margin-top:10px;">'
        f'<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">'
        f'<span style="font-size:12px; color:#3A4A66;">Key themes</span>'
        f'<span style="font-size:10px; padding:2px 8px; border-radius:20px; background:#1A2744; color:#6B9FD4; border:0.5px solid #1E3160;">{badge}</span>'
        f'</div>'
        f'<div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">'
        f'{topic_html}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )   

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Transcripts analysed", "30+")
c2.metric("Companies covered", "10")
c3.metric("NLP model", "FinBERT")


