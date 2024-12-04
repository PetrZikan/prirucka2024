import os
import pickle
import pytest
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from prirucka2024.fill_vector_store import fill_vector_store


class MockEmbeddings(Embeddings):
    """Mock embedding class to avoid external API calls."""

    def embed_documents(self, texts):
        """Generate simple deterministic embeddings based on text length."""
        return [[len(text)] for text in texts]

    def embed_query(self, text):
        """Generate a simple deterministic embedding for a query."""
        return [len(text)]


@pytest.fixture
def setup_test_data(tmp_path):
    """Set up test data for Chroma vector store."""
    test_pickle = tmp_path / "test_data.pkl"

    # Create dummy documents using langchain's Document class
    test_documents = [
        Document(
            metadata={"Header": "Header 1"}, page_content="Content under Header 1"
        ),
        Document(
            metadata={"Header": "Header 2"}, page_content="Content under Header 2"
        ),
    ]

    # Serialize documents to pickle
    with open(test_pickle, "wb") as f:
        pickle.dump(test_documents, f)

    return test_pickle


def test_fill_vector_store(tmp_path, setup_test_data):
    """Test the creation of a Chroma vector store using a mock embedding model."""
    chroma_db_dir = tmp_path / "chroma_db"

    # Use the mock embedding model
    fill_vector_store(
        pickle_file=str(setup_test_data),
        chroma_db_dir=str(chroma_db_dir),
        embedding_model=MockEmbeddings(),
    )

    # Verify the Chroma database directory was created
    assert chroma_db_dir.exists()

    # Load the persisted Chroma vector store
    vectorstore = Chroma(
        persist_directory=str(chroma_db_dir), embedding_function=MockEmbeddings()
    )

    # Check the number of documents in the vector store
    # assert vectorstore.index.docstore.count() == 2
    assert len(vectorstore.get()["ids"]) == 2


def test_fill_vector_store_empty_data(tmp_path):
    """Test behavior when the pickle file contains empty data."""
    empty_pickle = tmp_path / "empty_data.pkl"
    chroma_db_dir = tmp_path / "chroma_db"

    # Serialize an empty list to the pickle file
    with open(empty_pickle, "wb") as f:
        pickle.dump([], f)

    # Run the function
    fill_vector_store(
        pickle_file=str(empty_pickle),
        chroma_db_dir=str(chroma_db_dir),
        embedding_model=MockEmbeddings(),
    )

    # Verify the Chroma database directory was created
    assert chroma_db_dir.exists()

    # Load the persisted Chroma vector store
    vectorstore = Chroma(
        persist_directory=str(chroma_db_dir), embedding_function=MockEmbeddings()
    )

    # Check the number of documents in the vector store (should be 0)
    # assert vectorstore.index.docstore.count() == 0
    assert len(vectorstore.get()["ids"]) == 0


def test_fill_vector_store_missing_file(tmp_path):
    """Test behavior when the pickle file is missing."""
    missing_file = tmp_path / "missing_data.pkl"
    chroma_db_dir = tmp_path / "chroma_db"

    # Expect the function to raise a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        fill_vector_store(
            pickle_file=str(missing_file),
            chroma_db_dir=str(chroma_db_dir),
            embedding_model=MockEmbeddings(),
        )
