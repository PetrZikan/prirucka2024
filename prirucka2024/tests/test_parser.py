from prirucka2024.parser import extract_rst_files, parse_rst_file, parse_all_docs
from pathlib import Path

DOCS_DIR = Path("cadquery/doc")  # Path to the documentation directory
INDEX_FILE = DOCS_DIR / "index.rst"


def test_extract_rst_files():
    """Test that .rst files are correctly identified from index.rst."""
    rst_files = extract_rst_files()
    assert len(rst_files) > 0, "No .rst files were found."
    for file_path in rst_files:
        assert file_path.exists(), f"File does not exist: {file_path}"


def test_parse_rst_file():
    """Test parsing of a single .rst file."""
    rst_files = extract_rst_files()
    first_rst_file = rst_files[0]
    content = parse_rst_file(first_rst_file)
    assert isinstance(content, str), "Parsed content is not a string."
    assert len(content) > 0, "Parsed content is empty."


def test_parse_all_docs():
    """Test parsing of all relevant .rst files."""
    docs_content = parse_all_docs()
    assert isinstance(docs_content, dict), "Result is not a dictionary."
    assert len(docs_content) > 0, "No documents were parsed."

    cadquery_parsed_dir = Path("cadquery_parsed")
    cadquery_parsed_dir.mkdir(exist_ok=True)

    for file_name, content in docs_content.items():

        with open(cadquery_parsed_dir / file_name, "w") as f:
            f.write(content)

        assert isinstance(content, str), f"Content of {file_name} is not a string."
        assert len(content) > 0, f"Content of {file_name} is empty."
