from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat = ChatOpenAI(temperature=0.5)
messages = [
    SystemMessage(
        content="""Act as a senior software engineer at a startup company."""
    ),
    HumanMessage(
        content="""Please can you provide a funny joke about software engineers?"""
    ),
]

# for chunk in chat.stream(messages):
#     print(chunk.content, end="", flush=True)

response = chat.invoke(input=messages)
print(response.content)
