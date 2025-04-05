from typing import List
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import re


class SparseRetriever:
    def __init__(self) -> None:
        pass

    def query(self, query, num_pages=5) -> List[str]:
        results = []
        search_results = search(query, num_results=num_pages, lang='en')
        for url in search_results:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text()
                text = re.sub(r'\s+', ' ', text).strip()
                if text and len(text) > 20:
                    results.append(text)
        return results


if __name__ == "__main__":
    sr = SparseRetriever()
    results = sr.query("who is the first programmer")
    for r in results:
        print(r)
