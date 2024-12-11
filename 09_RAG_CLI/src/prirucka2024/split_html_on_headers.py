import os
import pickle
import logging
from rich import print
from langchain_text_splitters import HTMLHeaderTextSplitter

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define the headers for splitting
HEADERS_TO_SPLIT_ON = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(HEADERS_TO_SPLIT_ON)

from bs4 import BeautifulSoup


def preprocess_html(html_string, target_tag=None, target_class=None):
    """
    Preprocess HTML content by optionally focusing on a specific tag and class.

    Args:
        html_string (str): The raw HTML content.
        target_tag (str): The HTML tag to target (e.g., "div").
        target_class (str): The class of the target tag (e.g., "theme-doc-markdown markdown").

    Returns:
        str: Processed HTML content within the specified tag and class.
    """
    soup = BeautifulSoup(html_string, "html.parser")

    # If target_tag and target_class are specified, focus only on that section
    if target_tag and target_class:
        logger.info(f"Filtering content within <{target_tag} class='{target_class}'>")
        section = soup.find(target_tag, class_=target_class)
        if section:
            return str(section)
        else:
            logger.warning(
                f"No matching <{target_tag} class='{target_class}'> found in the HTML."
            )
            return ""  # Return an empty string if the tag is not found

    # Default preprocessing if no target is specified
    logger.info("No specific target specified; processing full HTML content.")
    return html_string


def split_html(
    file_path,
    output_pkl,
    output_txt,
    interactive,
    target_tag=None,
    target_class=None,
    drop_empty_metadata=True,
):
    """
    Split HTML file on headers and save results.

    Args:
        file_path (str): Path to the HTML file.
        output_pkl (str): Name of the pickle file for serialized splits.
        output_txt (str): Name of the text file for saving split contents.
        interactive (bool): Enable interactive mode for rejecting splits.
        target_tag (str): The HTML tag to target (e.g., "div").
        target_class (str): The class of the target tag (e.g., "theme-doc-markdown markdown").
        drop_empty_metadata (bool): Whether to drop splits with empty metadata.
    """
    try:
        # Read HTML file
        with open(file_path, "r", encoding="utf-8") as f:
            html_string = f.read()

        # Preprocess HTML content
        preprocessed_html = preprocess_html(html_string, target_tag, target_class)

        if not preprocessed_html.strip():
            logger.warning("Preprocessed HTML is empty. No splits generated.")
            with open(output_pkl, "wb") as f:
                pickle.dump([], f)
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write("")
            return

        # Use the preprocessed HTML for splitting
        all_html_header_splits = []
        # html_header_splits = html_splitter.split_text(preprocessed_html)
        html_header_splits = split_text_from_file(file_path) # see https://github.com/langchain-ai/langchain/issues/13149

        for split in html_header_splits:
            if drop_empty_metadata and not split.metadata:
                logger.info(f"Dropping split with empty metadata: {split.page_content}")
                continue

            if interactive:
                os.system("cls" if os.name == "nt" else "clear")  # Clear the screen
                print("=====")
                print(split.metadata)
                print(split.page_content)
                user_input = input(
                    "Press <Enter> to keep this split, <d> to disregard: "
                )
                if user_input.lower() == "d":
                    continue

            all_html_header_splits.append(split)

        # Save splits to pickle
        with open(output_pkl, "wb") as f:
            pickle.dump(all_html_header_splits, f)
        logger.info(f"Serialized splits saved to {output_pkl}")

        # Save splits to text file
        with open(output_txt, "w", encoding="utf-8") as f:
            for header_split in all_html_header_splits:
                f.write("=====\n")
                f.write(str(header_split.metadata) + "\n")
                f.write(header_split.page_content + "\n")
        logger.info(f"Split contents saved to {output_txt}")

        logger.info(f"Number of header splits: {len(all_html_header_splits)}")

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def split_text_from_file(file):
    """Split HTML file using BeautifulSoup.
    see: https://github.com/langchain-ai/langchain/issues/13149#issuecomment-2441237701

    Args:
        file: HTML file path or file-like object.

    Returns:
        List of Document objects with page_content and metadata.
    """
    from bs4 import BeautifulSoup
    from langchain.docstore.document import Document
    import bs4

    # Read the HTML content from the file or file-like object
    if isinstance(file, str):
        with open(file, 'r', encoding='utf-8') as f:
            html_content = f.read()
    else:
        # Assuming file is a file-like object
        html_content = file.read()

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract the header tags and their corresponding metadata keys
    headers_to_split_on = [tag[0] for tag in HEADERS_TO_SPLIT_ON]
    header_mapping = dict(HEADERS_TO_SPLIT_ON)

    documents = []

    # Find the body of the document
    body = soup.body if soup.body else soup

    # Find all header tags in the order they appear
    all_headers = body.find_all(headers_to_split_on)

    # If there's content before the first header, collect it
    first_header = all_headers[0] if all_headers else None
    if first_header:
        pre_header_content = ''
        for elem in first_header.find_all_previous():
            if isinstance(elem, bs4.Tag):
                text = elem.get_text(separator=' ', strip=True)
                if text:
                    pre_header_content = text + ' ' + pre_header_content
        if pre_header_content.strip():
            documents.append(Document(
                page_content=pre_header_content.strip(),
                metadata={}  # No metadata since there's no header
            ))
    else:
        # If no headers are found, return the whole content
        full_text = body.get_text(separator=' ', strip=True)
        if full_text.strip():
            documents.append(Document(
                page_content=full_text.strip(),
                metadata={}
            ))
        return documents

    # Process each header and its associated content
    for header in all_headers:
        current_metadata = {}
        header_name = header.name
        header_text = header.get_text(separator=' ', strip=True)
        current_metadata[header_mapping[header_name]] = header_text

        # Collect all sibling elements until the next header of the same or higher level
        content_elements = []
        for sibling in header.find_next_siblings():
            if sibling.name in headers_to_split_on:
                # Stop at the next header
                break
            if isinstance(sibling, bs4.Tag):
                content_elements.append(sibling)

        # Get the text content of the collected elements
        current_content = ''
        for elem in content_elements:
            text = elem.get_text(separator=' ', strip=True)
            if text:
                current_content += text + ' '

        # Create a Document if there is content
        if current_content.strip():
            documents.append(Document(
                page_content=current_content.strip(),
                metadata=current_metadata.copy()
            ))
        else:
            # If there's no content, but we have metadata, still create a Document
            documents.append(Document(
                page_content='',
                metadata=current_metadata.copy()
            ))

    return documents
