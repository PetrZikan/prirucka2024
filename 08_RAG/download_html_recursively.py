import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def download_page(url, output_dir):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        parsed_url = urlparse(url)
        page_name = parsed_url.path.strip("/").replace("/", "_") or "index"
        file_path = os.path.join(output_dir, page_name)

        # Save the page content
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)
        print(f"Downloaded: {url} -> {file_path}")
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return None


def extract_links(html_content, base_url):
    soup = BeautifulSoup(html_content, "html.parser")
    links = set()
    for a_tag in soup.find_all("a", href=True):
        link = urljoin(base_url, a_tag["href"])
        # Only keep links on the same domain and avoid fragments
        if urlparse(link).netloc == urlparse(base_url).netloc:
            links.add(link.split("#")[0])
    return links


def crawl_and_download(start_url, output_dir, visited=None):
    if visited is None:
        visited = set()

    if start_url in visited:
        return

    print(f"Crawling: {start_url}")
    visited.add(start_url)

    # Download the page and extract links
    html_content = download_page(start_url, output_dir)
    if html_content:
        links = extract_links(html_content, start_url)
        for link in links:
            crawl_and_download(link, output_dir, visited)


# Example usage:
start_url = "https://python.langchain.com/docs/tutorials/rag/"
output_dir = "downloaded_content"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

crawl_and_download(start_url, output_dir)
