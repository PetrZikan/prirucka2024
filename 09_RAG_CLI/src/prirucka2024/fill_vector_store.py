from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings

import logging
import pickle
from langchain_chroma import Chroma


def fill_vector_store(
    pickle_file, chroma_db_dir, embedding_model="text-embedding-ada-002"
):
    """
    Load serialized documents and populate a Chroma vector store.

    Args:
        pickle_file (str): Path to the pickle file containing serialized documents.
        chroma_db_dir (str): Directory to store the Chroma database.
        embedding_model (str or Embeddings): OpenAI embedding model name or an embedding instance.
    """
    try:
        # Load the serialized list of documents
        with open(pickle_file, "rb") as f:
            all_html_header_splits = pickle.load(f)

        if isinstance(embedding_model, str):
            embedding_instance = OpenAIEmbeddings(model=embedding_model)
        elif isinstance(embedding_model, Embeddings):
            embedding_instance = embedding_model
        else:
            raise ValueError(
                "Invalid embedding_model: must be a string or an Embeddings instance."
            )

        # Initialize and persist the vector store
        vectorstore = Chroma.from_documents(
            documents=all_html_header_splits,
            embedding=embedding_instance,
            persist_directory=chroma_db_dir,
        )
        # vectorstore.persist()
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        logging.error(f"An error occurred: {e}")
