import spacy, re
import gc
import streamlit as st

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
        _nlp.max_length = 2000000
    return _nlp


def get_doc(text):
    doc = get_nlp()(text)
    return doc

def process_text(text:str) -> dict:
    text = text[:30000]
    gc.collect()
    doc = get_nlp()(text)

    tokens= [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]

    entities=[(ent.text, ent.label_, ent.start) for ent in doc.ents if ent.label_ in ['ORG', 'MONEY', 'PERCENT', 'DATE', 'GPE']]

    Key_sentences=[sent.text.strip() for sent in doc.sents if any(phrase in sent.text.lower() for phrase in ['we expect',
            'we anticipate',
            'we forecast',
            'guidance',
            'outlook',
            'going forward',
            'next quarter',
            'full year']) and len(sent.text.split()) > 15]
    sentences=[sent.text.strip() for sent in doc.sents]
    result={'tokens': tokens, 'entities':entities,'key_sentences':Key_sentences,'sentences':sentences,'word_count':len(tokens), 'doc':doc, 'raw_text':text, 'sentence_count':len(sentences)}
    gc.collect()
    return result

def extract_eps_from_text(doc) -> float | None:
    eps_pattern = re.compile(
        r'(?:'
        r'(?:eps|earnings per share|diluted earnings per share|basic earnings per share)'
        r'\s*(?:--|—|-|of|was|were|is|:)?\s*'
        r'\$?\s*([-]?[\d]+\.[\d]+)'
        r'|'
        r'\$\s*([-]?[\d]+\.[\d]+)\s*(?:per share|basic|diluted|eps)'
        r')',
        re.IGNORECASE
    )

    for sent in doc.sents:
        sent_lower = sent.text.lower()

        if 'adjusted' in sent_lower:
            continue

        # truncate at comparison phrases
        text = sent.text
        for cutoff in ['compared to', 'prior year', 'prior-year', 'versus', 'vs.']:
            idx = text.lower().find(cutoff)
            if idx != -1:
                text = text[:idx]

        match = eps_pattern.search(text)
        if match:
            try:
                raw       = match.group(1) or match.group(2)
                eps_value = float(raw)
                if -50 < eps_value < 50:
                    return eps_value
            except (ValueError, TypeError):
                continue

    return None 