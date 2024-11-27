from rich import print
import requests


def download_html(url, output_file):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(response.text)
        print(f"HTML content saved to {output_file}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


# Example usage:
url = "https://python.langchain.com/docs/tutorials/rag/"
output_file = "langchain_rag.html"
download_html(url, output_file)
