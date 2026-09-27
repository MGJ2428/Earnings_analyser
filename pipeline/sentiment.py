import re


def score_sentences(sentences: list, finbert) -> list[dict]:
    if not sentences:
        return []
    valid = [s for s in sentences if len(s.split()) > 5]
    valid = [s[:1000] for s in valid]
    if not valid:
        return []
    results = []
    batch_size = 16
    for i in range(0, len(valid), batch_size):
        batch = valid[i:i + batch_size]
        try:
            batch_results = finbert(batch)
            results.extend(batch_results)
        except Exception:
            continue
    return results


def aggregate_sentiment(results: list[dict]) -> dict:
    if not results:
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0,'overall': 0.0, 'label': 'neutral', 'sentence_count': 0}
    counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for r in results:
        counts[r['label'].lower()] += 1
    total = len(results)
    pos_prop = round(counts['positive'] / total, 2)
    neg_prop = round(counts['negative'] / total, 2)
    neu_prop = round(counts['neutral']  / total, 2)
    overall_score = round(pos_prop - neg_prop, 2)
    if overall_score > 0.1:
        label = 'positive'
    elif overall_score < -0.1:
        label = 'negative'
    else:
        label = 'neutral'
    return {'positive':pos_prop,'negative':neg_prop,'neutral': neu_prop,'overall':overall_score,'label':label,'sentence_count': total}


def segment_transcript(text: str, sentences: list) -> dict:
    full_text_lower = text.lower()
    qa_marks = ['question and answer', 'q&a', 'open the call', 'open for questions', 'first question']
    cfo_ids = ['chief financial officer', 'cfo', 'our cfo', 'financial results', 'turning to our financials']
    pos_of_qs = len(text)
    for marker in qa_marks:
        pos = full_text_lower.find(marker)
        if pos != -1 and pos < pos_of_qs:
            pos_of_qs = pos
    cfo_pos = len(text) // 3
    for marker in cfo_ids:
        pos = full_text_lower.find(marker)
        if pos != -1:
            cfo_pos = pos
            break
    ceo_sentences = []
    cfo_sentences = []
    qa_sentences = []
    current_pos = 0
    for sent in sentences:
        sent_pos = text.find(sent, current_pos)
        if sent_pos == -1:
            sent_pos = current_pos
        if sent_pos >= pos_of_qs:
            qa_sentences.append(sent)
        elif sent_pos >= cfo_pos:
            cfo_sentences.append(sent)
        else:
            ceo_sentences.append(sent)
        current_pos = max(current_pos, sent_pos)
    return {'ceo': ceo_sentences, 'cfo': cfo_sentences, 'qa': qa_sentences}

def extract_speakers_from_intro(text: str) -> dict:
    intro_text = text[:3000]
    joining_pattern = re.compile(r'(?:joining me today are|on the call today are|with me today are|'r'joining us today are|we have with us)\s+(.+?)(?:\.|and we)',re.IGNORECASE | re.DOTALL)
    match = joining_pattern.search(intro_text)
    if match:
        names_text = match.group(1)
        name_pattern = re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)')
        return {name: 'other' for name in name_pattern.findall(names_text) if len(name.split()) >= 2}
    return {}

def segment_by_speaker(text: str, sentences: list) -> dict:
    speaker_pattern = re.compile(r'^([A-Z][a-zA-Z\s\.\-]{3,40}):\s', re.MULTILINE)
    participant_pattern = re.compile(r'-\s*([^—\-\n]+?)\s*[—\-]\s*([^\n]+)', re.MULTILINE)

    participants = {}
    for match in participant_pattern.finditer(text[:3000]):
        title = match.group(1).strip()
        name = match.group(2).strip()
        participants[name] = title
        
    if len(participants) < 2:
        intro_speakers = extract_speakers_from_intro(text)
        for name in intro_speakers:
            if name not in participants:
                participants[name] = ''
    noise_names = {
        'image source', 'industry glossary', 'read next',
        'full conference call transcript', 'call participants',
        'takeaways', 'risks', 'summary', 'stocks mentioned'}

    turns = []
    for match in speaker_pattern.finditer(text):
        name = match.group(1).strip()
        if '\n' in name:
            continue
        if name.lower() in noise_names:
            continue
        if len(name.split()) > 5:
            continue
        pos = match.start()
        title = ''
        for p_name, p_title in participants.items():
            p_last = p_name.strip().split()[-1].lower()
            s_last = name.strip().split()[-1].lower()
            if (name.lower() in p_name.lower() or p_name.lower() in name.lower() or s_last == p_last):
                title = p_title
                break
        title_lower = title.lower()
        name_lower  = name.lower()
        if any(t in title_lower for t in ['chief executive', 'ceo', 'president', 'incoming']):
            role = 'ceo'
        elif any(t in title_lower for t in ['chief financial officer', 'cfo', 'finance']):
            role = 'cfo'
        elif 'operator' in name_lower:
            role = 'operator'
        elif 'investor relations' in title_lower or 'director' in title_lower:
            role = 'ir'
        elif any(t in title_lower for t in ['analyst', 'research', 'securities', 'capital', 'bank']):
            role = 'analyst'
        else:
            role = 'other'
        turns.append({'name': name, 'title': title, 'role': role, 'pos': pos})

    ceo_sentences = []
    cfo_sentences = []
    qa_sentences = []
    speaker_sentences = {}
    speaker_roles = {}
    current_pos = 0

    for sent in sentences:
        sent_pos = text.find(sent, current_pos)
        if sent_pos == -1:
            sent_pos = current_pos
        active_role = 'other'
        active_name = 'unknown'
        for turn in turns:
            if turn['pos'] <= sent_pos:
                active_role = turn['role']
                active_name = turn['name']
            else:
                break
        if active_role == 'ceo':
            ceo_sentences.append(sent)
        elif active_role == 'cfo':
            cfo_sentences.append(sent)
        elif active_role == 'analyst':
            qa_sentences.append(sent)
        if active_name not in speaker_sentences:
            speaker_sentences[active_name] = []
            speaker_roles[active_name] = active_role
        speaker_sentences[active_name].append(sent)
        current_pos = max(current_pos, sent_pos)

    return {'ceo':ceo_sentences,'cfo':cfo_sentences,'qa':qa_sentences,'speaker_sentences': speaker_sentences,'speaker_roles': speaker_roles,'turns':turns}


def analyse_transcript_sentiment(text: str, sentences: list, finbert) -> dict:
    segments = segment_by_speaker(text, sentences)

    fallback = False
    if not segments['ceo'] and not segments['cfo']:
        segments = segment_transcript(text, sentences)
        fallback = True

    ceo_results = score_sentences(segments['ceo'][:30], finbert)
    cfo_results = score_sentences(segments['cfo'][:20], finbert)
    qa_results  = score_sentences(segments['qa'][:20],  finbert)
    all_results = score_sentences(sentences[:60],        finbert)

    per_speaker = {}
    if not fallback:
        for name, sents in segments['speaker_sentences'].items():
            if name == 'unknown':
                continue
            if len(sents) < 2:
                continue
            role = segments.get('speaker_roles', {}).get(name, 'other')
            if role == 'operator':
                continue
            results = score_sentences(sents[:20], finbert)
            agg = aggregate_sentiment(results)
            agg['role']           = role
            agg['sentence_count'] = len(sents)
            per_speaker[name]     = agg

    return {'overall':aggregate_sentiment(all_results),'ceo':aggregate_sentiment(ceo_results),'cfo':aggregate_sentiment(cfo_results),'qa':aggregate_sentiment(qa_results),'per_speaker': per_speaker,'speakers':list(segments.get('speaker_sentences',{}).keys()),'segment_sizes': {'ceo': len(segments['ceo']),'cfo': len(segments['cfo']),'qa':  len(segments['qa'])}}