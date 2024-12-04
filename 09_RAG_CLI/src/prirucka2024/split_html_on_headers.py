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
        html_header_splits = html_splitter.split_text(preprocessed_html)

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
