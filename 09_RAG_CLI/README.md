# RAG CLI

This Python project was initiated by the following command:

```
poetry new . --src
```

# Installation
```
pip install -e .
```

# Test
```
prirucka2024 download-url https://python-poetry.org/docs/pyproject/ poetry.html
```

# Build a Chatbot
```
prirucka2024 download-url https://python.langchain.com/docs/tutorials/chatbot/ langchain_docs_build_a_chatbot.html
prirucka2024 split-html-on-headers langchain_docs_build_a_chatbot.html --output-pkl langchain_docs_build_a_chatbot.pkl --output-txt langchain_docs_build_a_chatbot.txt --drop-empty-metadata
prirucka2024 fill-vector-store langchain_docs_build_a_chatbot.pkl langchain_docs_build_a_chatbot_chroma
prirucka2024 rag --chroma-db-dir langchain_docs_build_a_chatbot_chroma/ "Provide detailed instruction on how to manage conversation history."
```
