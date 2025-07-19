# ingest.py
import os
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")

# Create the list of Documents (each with the ISBN as content)
documents = [
    Document(page_content=str(row["isbn13"]))
    for _, row in books.iterrows()
]

# Setup Chroma vector store
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")

db = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(),
    persist_directory=PERSIST_DIRECTORY
)

db.persist()

print(f"✅ Ingestion complete. {len(documents)} books added.")