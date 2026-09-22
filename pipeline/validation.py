from urllib.parse import urlparse

def validate_url(url:str) -> tuple[bool, str]:
    try:
        result=urlparse(url)

        if not all([result.scheme, result.netloc]):
            return False, "Invlid URL - make sure it starts with https://"
        if result.scheme not in ['https', 'https']:
            return False, 'Must start with http or https'
        if len(result.netloc)<4:
            return False, 'URL is not valid'
        return True, 'Valid URL'
    except Exception:
        return False, 'Could not parse URL'
    


