import glob
from langchain_text_splitters import HTMLHeaderTextSplitter
from rich import print
import logging
import pickle  # Add this import
import os  # Add this import

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)

# file_paths = glob.glob("downloaded_content/*.html")
file_paths = ["langchain_rag.html"]

all_html_header_splits = []
for file_path in file_paths:
    logger.info(f"Processing {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        html_string = f.read()

    html_header_splits = html_splitter.split_text(html_string)
    for split in html_header_splits:
        os.system("cls" if os.name == "nt" else "clear")  # Clear the screen
        print("=====")
        print(split.metadata)
        print(split.page_content)
        user_input = input("Press <Enter> to keep this split, <d> to disregard: ")
        if user_input.lower() != "d":
            all_html_header_splits.append(split)

# Serialize the list of documents
with open("all_html_header_splits.pkl", "wb") as f:
    pickle.dump(all_html_header_splits, f)

logger.info("Number of header splits: %d", len(all_html_header_splits))

with open("split_headers.txt", "w", encoding="utf-8") as f:
    for header_split in all_html_header_splits:
        f.write("=====\n")
        f.write(str(header_split.metadata) + "\n")
        f.write(header_split.page_content + "\n")
