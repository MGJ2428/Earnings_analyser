import trafilatura
from bs4 import BeautifulSoup

def extract_text(html:str) -> tuple[str | None, str | None]:
    text = trafilatura.extract(html, include_comments=False, include_tables=False, no_fallback=False)
    if text and len(text)>500:
        return text, None
    bs_output=BeautifulSoup(html, 'html.parser')
    for tag in bs_output(['nav', 'footer', 'header', 'script', 'style' ,'aside', 'advertisement']):
        tag.decompose()
    text=bs_output.get_text(separator=' ', strip=True)
    if text and len(text)>500:
        return text, None
    return None, 'Could not extract enough readable text from the page'

