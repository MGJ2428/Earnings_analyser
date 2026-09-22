import requests

def fetch_page(url:str) -> tuple[str | None , str | None]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    try:
        response=requests.get(url, headers= headers, timeout = 10)
        if response.status_code == 200:
            return response.text, None
        elif response.status_code == 404:
            return None, 'Page could not be found'
        elif response.status_code == 403:
            return None, 'Access denied - blocks scrapers.'
        else:
            return None, 'Failed to fetch page'
    except requests.Timeout:
        return None, 'Request timed out - check connection'
    except requests.ConnectionError:
        return None, "Could not connect - check URL"