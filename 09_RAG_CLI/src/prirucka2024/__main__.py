import click

from prirucka2024.download_url import download_url as download_url_func
from prirucka2024.rag import prompt, retrieve
from prirucka2024.split_html_on_headers import split_html as split_html_func
from prirucka2024.fill_vector_store import fill_vector_store as fill_vector_store_func
from prirucka2024.pdf_parser_raw import parse_pdf_raw as parse_pdf_raw_func
from prirucka2024.split_text_recursively import (
    split_text_recursively as split_text_recursively_func,
)

# from prirucka2024.pdf_parser import parse_pdf as parse_pdf_func


@click.group()
def main():
    """CLI for the prirucka2024 RAG application."""
    pass


@main.command()
@click.argument("url")
@click.argument("output_file")
def download_url(url, output_file):
    """Download the HTML content from a given URL and save it to a file."""
    download_url_func(url, output_file)


@main.command()
@click.argument("file_path")
@click.option(
    "--output-pkl",
    default="all_html_header_splits.pkl",
    help="Name of the pickle output file.",
)
@click.option(
    "--output-txt", default="split_headers.txt", help="Name of the text output file."
)
@click.option(
    "--interactive", is_flag=True, help="Enable interactive mode for rejecting splits."
)
@click.option(
    "--drop-empty-metadata", is_flag=True, help="Drop splits with empty metadata."
)
def split_html_on_headers(
    file_path, output_pkl, output_txt, interactive, drop_empty_metadata
):
    """Split HTML file on headers and save results."""
    split_html_func(file_path, output_pkl, output_txt, interactive, drop_empty_metadata)


@main.command()
@click.argument("pickle_file")
@click.argument("chroma_db_dir")
@click.option(
    "--embedding-model",
    default="text-embedding-ada-002",
    help="OpenAI embedding model to use for vectorization.",
)
def fill_vector_store(pickle_file, chroma_db_dir, embedding_model):
    """
    Load serialized documents and populate a Chroma vector store.
    """
    fill_vector_store_func(pickle_file, chroma_db_dir, embedding_model)


@main.command()
@click.argument("question")
@click.option(
    "--chroma-db-dir",
    required=True,
    help="Path to the Chroma database directory.",
)
@click.option(
    "--k",
    default=5,
    help="Number of top documents to retrieve (default: 5).",
)
@click.option(
    "--embedding-model",
    default="text-embedding-ada-002",
    help="OpenAI embedding model to use for retrieval.",
)
@click.option(
    "--llm-model",
    default="gpt-4o-mini",
    help="LLM model to use for generating the response.",
)
def rag(question, chroma_db_dir, k, embedding_model, llm_model):
    """
    Retrieve documents from Chroma and generate a response using LLM.
    """
    # Retrieve documents
    docs = retrieve(
        chroma_db_dir=chroma_db_dir,
        question=question,
        k=k,
        embedding_model=embedding_model,
    )

    # Generate response
    response = prompt(docs=docs, question=question, llm_model=llm_model)

    # Print results
    click.echo("Retrieved Documents:")
    for doc in docs:
        click.echo(f"Metadata: {doc.metadata}")
        click.echo(f"Content: {doc.page_content}")
        click.echo("=" * 40)

    click.echo("Generated Response:")
    click.echo(response)


@main.command()
@click.argument("infile")
@click.argument("outfile")
def parse_pdf_raw(infile, outfile):
    """Force-parse the text content of a PDF file and save it to a text file."""

    parse_pdf_raw_func(infile, outfile)


@main.command()
@click.argument("infile")
@click.argument("outfile")
@click.option(
    "--chunk-size", default=200, help="Size of the chunks to split the text into."
)
@click.option("--chunk-overlap", default=50, help="Size of the overlap between chunks.")
def split_text_recursively(infile, outfile, chunk_size, chunk_overlap):
    """Force-parse the text content of a PDF file and save it to a text file."""

    documents = split_text_recursively_func(infile, outfile, chunk_size, chunk_overlap)


# @main.command()
# @click.argument("infile")
# @click.argument("outfile")
# def parse_pdf(infile, outfile):
#    """Parse the text content of a PDF file and save it in a better fashion to a text file."""
#
#    parse_pdf_func(infile, outfile)


if __name__ == "__main__":
    main()
