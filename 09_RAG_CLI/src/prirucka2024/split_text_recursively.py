import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_recursively(infile, outfile, chunk_size=200, chunk_overlap=50):
    """
    Split the text content of a file recursively and save it to a text file.

    Args:
        infile (str): The file path to the text file to split.
        outfile (str): The file path to save the split text content to.
    """
    with open(infile) as f:
        content = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    # pickle the documents
    documents = text_splitter.create_documents([content])

    with open(outfile, "wb") as f:
        pickle.dump(documents, f)
