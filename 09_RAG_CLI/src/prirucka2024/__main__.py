import click

from prirucka2024.download_url import download_url as download_url_func


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


if __name__ == "__main__":
    main()
