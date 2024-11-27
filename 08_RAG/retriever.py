from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rich import print

CHROMA_DB_DIR = "./chroma_db_langchain"
question = "What is Langchain RAG?"

vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR, embedding_function=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

retrieved_docs = retriever.invoke(question)

for doc in retrieved_docs:
    print(doc.metadata)
    print(doc.page_content)
    print("=====")

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate(
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
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke(question)

print("**************************")
print(response)
