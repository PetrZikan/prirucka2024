import pickle
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DB_DIR = "./chroma_db_langchain"

# Load the serialized list of documents
with open("all_html_header_splits.pkl", "rb") as f:
    all_html_header_splits = pickle.load(f)

vectorstore = Chroma.from_documents(
    documents=all_html_header_splits,
    embedding=OpenAIEmbeddings(),
    persist_directory=CHROMA_DB_DIR,
)
