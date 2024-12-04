from rich import print
import requests


def download_url(url, output_file):
    """
    Download the HTML content from a given URL and save it to a file.

    Args:
        url (str): The URL to download the HTML content from.
        output_file (str): The file path to save the HTML content to.
    """

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(response.text)
        print(f"HTML content saved to {output_file}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
