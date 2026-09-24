import streamlit as st
from pipeline.predicted_earnings_checker import plot_earnings_history, get_past_all_earnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pipeline.correlation import calculate_correlations, plot_correlation
st.set_page_config(page_title='Results', layout='wide')

if 'nlp_results' not in st.session_state:
    st.warning('No report analysed.')
    if st.button('Go back'):
        st.switch_page('app.py')
    st.stop()

nlp_results = st.session_state['nlp_results']
earnings    = st.session_state.get('earnings')
ticker      = st.session_state.get('ticker')
source_url  = st.session_state.get('source_url')

st.title('Analysis of Report')

if source_url:
    st.caption(f'Source: {source_url}')
if ticker:
    st.caption(f'Company: {ticker}')

st.divider()

st.subheader("Earnings vs Analyst Expectations")
earnings = st.session_state.get('earnings')


if earnings and earnings.get('success'):
    st.caption(f"EPS source: {earnings.get('source', 'unknown')}   Quarter: {earnings.get('quarter', '')}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Analyst Estimate",  f"${earnings.get('estimated_eps', 'N/A')}")
    col2.metric(
        "Actual EPS",
        f"${earnings.get('actual_eps', 'N/A')}",
        f"{earnings.get('surprise', 0)}%"
    )
    verdict = earnings.get('verdict', '')
    if verdict == 'BEAT':
        col3.success(f"✅ BEAT by ${earnings.get('difference', '')}")
    elif verdict == 'MISS':
        col3.error(f"❌ MISSED by ${abs(earnings.get('difference', 0))}")
    else:
        col3.info("➡️ MET expectations")
else:
    st.info("Could not retrieve earnings data for this company.")
    

st.divider()
st.subheader("Sentiment Analysis")
st.caption("Powered by FinBERT — fine-tuned on financial text")

sentiment = st.session_state.get('nlp_results', {}).get('sentiment')

if sentiment:
    overall = sentiment['overall'].get('overall','0.0')
    label   = sentiment['overall'].get('label', 'neutral')

    if overall > 0.1:
        st.success(f"Overall tone: **Positive** ({overall:+.2f})")
    elif overall < -0.1:
        st.error(f"Overall tone: **Negative** ({overall:+.2f})")
    else:
        st.info(f"Overall tone: **Neutral** ({overall:+.2f})")

    sent_col1, sent_col2, sent_col3 = st.columns(3)

    per_speaker = sentiment.get('per_speaker', {})
    managment = {k: v for k, v in per_speaker.items() if v['role'] in ['ceo', 'cfo']}
    analysts = {k: v for k, v in per_speaker.items() if v['role'] == 'analyst'}
    if managment:
        st.markdown('Management speakers')
        cols=st.columns(min(len(managment),3))
        for i, (name, data) in enumerate(managment.items()):
            score = data.get('overall', 0.0)
            lbl = data.get('label', 'neutral').capitalize()
            n = data.get('sentence_count', 0)
            p = data.get('positive', 0.0)
            ne = data.get('negative', 0.0)
            nu = data.get('neutral', 0.0)
            with cols[i%len(cols)]:
                with st.container(border=True):
                    st.metric(name, f"{score:+.2f}", lbl)
                    st.caption(f"{n} sentences")
                    st.markdown(f"""
                        <div style="display:flex;gap:3px;margin-top:4px;">
                            <div style="flex:{max(p,0.01)};height:4px;background:#1baf7a;border-radius:2px;"></div>
                            <div style="flex:{max(nu,0.01)};height:4px;background:#6B7A99;border-radius:2px;"></div>
                            <div style="flex:{max(ne,0.01)};height:4px;background:#e34948;border-radius:2px;"></div>
                            </div>
                            """, unsafe_allow_html=True)

    if managment:
        mgmt_avg    = sum(v['overall'] for v in managment.values()) / len(managment)
        st.info(f"Management tone {mgmt_avg:+.2f}")
else:
    st.info("Sentiment data not available.")

st.divider()

st.subheader('Key Topics')
st.caption('Discovered using LDA topic modelling')
topics = st.session_state.get('nlp_results', {}).get('topics',[])
if topics:
    cols=st.columns(len(topics))
    for i , (topic, col) in enumerate(zip(topics, cols)):
        with col:
            with st.container(border=True):
                st.markdown(f"Topic - {topic['label']}")
else:
    st.info('Topic modelling not available.')


st.divider()

st.subheader("Forward Guidance")
key_sentences = st.session_state.get('nlp_results', {}).get('key_sentences', [])
if key_sentences:
    for sentence in key_sentences:
        with st.container(border=True):
            st.write(sentence)
else:
    st.info("No forward guidance statements found.")
from pipeline.correlation import calculate_correlations, plot_correlation
from scipy import stats as scipy_stats

st.divider()

st.subheader("Sentiment vs Return Correlation")
st.caption("Does CEO tone predict post-earnings stock movement?")

corr = calculate_correlations()

if corr.get('success'):
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall r",f"{corr['r']:+.2f}")
    col2.metric("Data points",corr['n'])
    col3.metric("Significant","Yes" if corr['significant'] else "No",f"p = {corr['p_value']}")

    if corr.get('beat_r') is not None or corr.get('miss_r') is not None:
        st.markdown("By earnings verdict")
        bc1, bc2 = st.columns(2)
        with bc1:
            with st.container(border=True):
                if corr.get('beat_r') is not None:
                    st.metric("Beat quarters",f"r = {corr['beat_r']:+.2f}",f"{corr['beat_n']} quarters")
                    st.caption("Sentiment correlation within beat quarters")
                else:
                    st.caption("Need 3+ beat quarters")
        with bc2:
            with st.container(border=True):
                if corr.get('miss_r') is not None:
                    st.metric("Miss quarters",f"r = {corr['miss_r']:+.2f}",f"{corr['miss_n']} quarters")
                    st.caption("Sentiment correlation within miss quarters")
                else:
                    st.caption("Need 3+ miss quarters")

    if corr['r'] > 0.3:
        st.success("Positive sentiment tends to predict positive post-earnings returns.")
    elif corr['r'] < -0.3:
        st.warning("Positive sentiment precedes negative returns — the market may over-price optimism.")
    else:
        st.info("No strong relationship detected yet — analyse more transcripts.")
    current_ticker = st.session_state.get('ticker')
    current_date   = st.session_state.get('call_date')

    fig = plot_correlation(corr['data'],highlight_ticker = current_ticker,highlight_date   = current_date
)
    fig = plot_correlation(corr['data'])
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)

else:
    st.info(corr.get('error', 'Not enough data yet.'))
    if corr.get('n'):
        st.caption(f"Have {corr['n']} data point(s) — need 5 to calculate correlation.")

st.divider()
st.subheader("Earnings History")

past = st.session_state.get('nlp_results', {}).get('past_earnings')

if past:
    quarters = past['quarters']
    estimates = past['estimates']
    actuals = past['actuals']
    beats = [a >= e for a, e in zip(actuals, estimates)]
    colours = ['#1baf7a' if b else '#e34948' for b in beats]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(quarters, estimates, color='#2a78d6', linestyle='--',
            marker='s', markersize=7, label='Analyst estimate')
    for q, actual, colour in zip(quarters, actuals, colours):
        ax.scatter(q, actual, color=colour, s=120, zorder=5,
                   edgecolors='white', linewidths=1.5)
    ax.set_ylabel('EPS ($)', fontsize=12)
    ax.set_xlabel('Quarter', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', color='#e1e0d9', linewidth=0.8)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    beat_patch = mpatches.Patch(color='#1baf7a', label='Beat')
    miss_patch = mpatches.Patch(color='#e34948', label='Miss')
    est_patch  = mpatches.Patch(color='#2a78d6', label='Analyst estimate')
    ax.legend(handles=[beat_patch, miss_patch, est_patch], loc='upper left')
    plt.tight_layout()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)
else:
    st.info("No earnings history available.")

if st.button("← Analyse another report"):
    for key in ['nlp_results', 'earnings', 'ticker', 'source_url', 'raw_text', 'needs_ticker']:
        st.session_state.pop(key, None)
    st.switch_page("app.py")


