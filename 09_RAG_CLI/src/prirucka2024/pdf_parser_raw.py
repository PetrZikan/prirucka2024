from langchain_community.document_loaders import PyPDFLoader


def parse_pdf_raw(infile, outfile):
    """
    Parse the text content of a PDF file and save it to a text file.

    Args:
        infile (str): The file path to the PDF file to parse.
        outfile (str): The file path to save the parsed text content to.
    """

    try:
        loader = PyPDFLoader(infile)
        documents = loader.load()
        with open(outfile, "w", encoding="utf-8") as file:
            for document in documents:
                file.write(document.page_content + "\n")

        print(f"PDF content saved to {outfile}")
    except Exception as e:
        print(f"An error occurred while parsing the PDF file: {e}")
