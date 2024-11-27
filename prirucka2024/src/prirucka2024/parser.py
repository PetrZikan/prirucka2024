from pathlib import Path
from bs4 import BeautifulSoup

DOCS_DIR = Path("cadquery/doc")  # Path to the documentation directory
INDEX_FILE = DOCS_DIR / "index.rst"


def extract_rst_files():
    """
    Parse the index.rst file to identify referenced .rst files.
    Returns:
        list: Paths to the .rst files listed in the toctree.
    """
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Index file not found: {INDEX_FILE}")

    rst_files = []
    with open(INDEX_FILE, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip().endswith(".rst"):
                rst_files.append(DOCS_DIR / line.strip())
    return rst_files


def parse_rst_file(file_path):
    """
    Parse the content of a single .rst file.
    Args:
        file_path (Path): Path to the .rst file.
    Returns:
        str: Cleaned text content of the file.
    """
    with open(file_path, "r") as f:
        content = f.read()

    # Clean up the content (remove directives, links, etc.)
    soup = BeautifulSoup(content, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text.strip()


def parse_all_docs():
    """
    Parse all relevant .rst files and return their contents.
    Returns:
        dict: A dictionary where keys are file names and values are their text content.
    """
    rst_files = extract_rst_files()
    docs_content = {}
    for rst_file in rst_files:
        docs_content[rst_file.name] = parse_rst_file(rst_file)
    return docs_content
