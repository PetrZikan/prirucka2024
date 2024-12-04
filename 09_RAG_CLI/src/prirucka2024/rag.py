from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def retrieve(chroma_db_dir, question, k=5, embedding_model="text-embedding-ada-002"):
    """
    Retrieve the top k documents from the Chroma database.

    Args:
        chroma_db_dir (str): Path to the Chroma database directory.
        question (str): The question for which to retrieve relevant documents.
        k (int): Number of top documents to retrieve.
        embedding_model (str or Embeddings): OpenAI embedding model or custom embedding instance.

    Returns:
        list: A list of retrieved documents.
    """
    try:
        # Initialize Chroma vector store
        if isinstance(embedding_model, str):
            embedding_instance = OpenAIEmbeddings(model=embedding_model)
        elif isinstance(embedding_model, Embeddings):
            embedding_instance = embedding_model
        else:
            raise ValueError(
                "Invalid embedding_model: must be a string or an Embeddings instance."
            )

        vectorstore = Chroma(
            persist_directory=chroma_db_dir, embedding_function=embedding_instance
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )

        # Retrieve documents
        retrieved_docs = retriever.invoke(question)
        logger.info(f"Retrieved {len(retrieved_docs)} documents.")
        return retrieved_docs
    except Exception as e:
        logger.error(f"An error occurred during retrieval: {e}")
        raise


def prompt(docs, question, llm_model="gpt-4o-mini"):
    """
    Generate a response using the retrieved documents and a question.

    Args:
        docs (list): A list of retrieved documents.
        question (str): The question to answer.
        llm_model (str): The LLM model to use for generating the response.

    Returns:
        str: The generated response.
    """
    try:
        llm = ChatOpenAI(model=llm_model)

        # Define the prompt template
        chat_prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. 

Question: {question} 
Context: {context} 

Answer:""",
                )
            ]
        )

        def format_docs(docs):
            return "\n\n".join(docs)

        # Define the RAG chain
        rag_chain = (
            {
                "context": RunnablePassthrough() | format_docs,
                "question": RunnablePassthrough(),
            }
            | chat_prompt
            | llm
            | StrOutputParser()
        )

        # Generate response
        response = rag_chain.invoke({"context": docs, "question": question})
        return response
    except Exception as e:
        logging.error(f"An error occurred during prompting: {e}")
        raise
