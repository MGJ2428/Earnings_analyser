import requests
from gensim import corpora
from gensim.models import LdaModel
from collections import Counter

STOPWORDS = {
    'year', 'quarter', 'company', 'business', 'percent',
    'million', 'billion', 'share', 'result', 'report',
    'period', 'prior', 'versus', 'compared', 'basis',
    'total', 'increase', 'decrease', 'growth', 'rate',
    'second', 'first', 'third', 'fourth', 'half',
    'operating', 'financial', 'revenue', 'income',
    'thank', 'please', 'call', 'today', 'good',
    'know', 'think', 'want', 'going', 'come',
    'question', 'answer', 'comment', 'provide',
    'strong', 'well', 'also', 'great', 'continue',
    'adjust', 'adjusted', 'ebitda', 'gaap', 'currency',
    'constant', 'impact', 'look', 'make', 'take',
    'give', 'back', 'next', 'last', 'point', 'term',
    'level', 'number', 'current', 'really', 'just',
    'right', 'okay', 'actually', 'certainly', 'absolutely',
    'obviously', 'exactly', 'basically', 'generally', 'largely',
    'slightly', 'primarily', 'approximately', 'roughly',
    'conclude', 'color', 'operator', 'participant', 'line',
    'morning', 'afternoon', 'welcome', 'standing', 'conference',
    'speaker', 'hand', 'turn', 'remark', 'prepared', 'open',
    'floor', 'expect', 'guidance', 'reflect', 'date',
    'report', 'discuss', 'highlight', 'mention', 'note',
    'include', 'consider', 'believe', 'remain', 'represent',
    'follow', 'drive', 'support', 'allow', 'enable',
    'achieve', 'reach', 'target', 'deliver', 'execute'
}


def label_topic_with_llm(words: list) -> str:
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 20,
                "messages": [{
                    "role": "user",
                    "content": (
                        f"These words come from topic modelling an earnings call transcript: "
                        f"{', '.join(words)}. "
                        f"Summarise what business theme these words represent in 2-3 words maximum. "
                        f"Reply with ONLY the label, nothing else. "
                        f"Examples: 'AI Strategy', 'Supply Chain', 'Revenue Growth', "
                        f"'Geographic Expansion', 'Student Retention', 'Margin Expansion'."
                    )
                }]
            },
            timeout=10
        )
        data = response.json()
        label = data['content'][0]['text'].strip().strip('."\'').strip()
        return label
    except Exception:
        return words[0].capitalize() if words else 'General'


def extract_topics(tokens: list, num_topics: int = 4,
                   company_name: str = '',
                   extra_stops: set = None) -> list[dict]:

    if len(tokens) < 50:
        return []

    dynamic_stops = STOPWORDS.copy()

    if company_name:
        for word in company_name.lower().split():
            if len(word) > 2:
                dynamic_stops.add(word)

    if extra_stops:
        dynamic_stops.update(extra_stops)

    # filter words appearing more than 1.5% of tokens
    word_counts = Counter(tokens)
    total = len(tokens)
    very_common = {
        w for w, c in word_counts.items()
        if c / total > 0.015
    }
    dynamic_stops.update(very_common)

    filtered = [
        t for t in tokens
        if t not in dynamic_stops
        and len(t) > 3
        and not t.isdigit()
    ]

    if len(filtered) < 30:
        return []

    chunk_size = 40
    chunks = [
        filtered[i:i + chunk_size]
        for i in range(0, len(filtered), chunk_size)
        if len(filtered[i:i + chunk_size]) > 8
    ]

    if len(chunks) < 4:
        return []

    dictionary = corpora.Dictionary(chunks)
    dictionary.filter_extremes(no_below=2, no_above=0.6)

    if len(dictionary) < 10:
        return []

    corpus = [dictionary.doc2bow(chunk) for chunk in chunks]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=20,
        alpha='auto',
        eta='auto',
        random_state=42
    )

    topics = []
    for i in range(num_topics):
        top_words = [word for word, _ in lda.show_topic(i, topn=5)]
        label     = label_topic_with_llm(top_words)
        topics.append({
            'id':    i + 1,
            'words': top_words,
            'label': label
        })

    return topics