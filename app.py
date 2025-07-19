import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables (e.g., for OpenAI API key)
load_dotenv()

# --- Setup FastAPI ---
app = FastAPI()

# Allow frontend (e.g., Wix) to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Chroma DB and Book Dataset ---
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")
books = pd.read_csv("books_with_emotions.csv")  # CSV must be in repo

db_books = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=OpenAIEmbeddings()
)

print(f"Chroma DB loaded from: {PERSIST_DIRECTORY}")


# --- Request Body Schema ---
class RecommendationRequest(BaseModel):
    query: str
    category: str = "All"
    tone: str = None


# --- Recommendation Logic ---
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs


# --- API Endpoint ---
@app.post("/recommend")
def recommend_books(req: RecommendationRequest):
    print(f"Received: {req}")
    try:
        recommendations = retrieve_semantic_recommendations(req.query, req.category, req.tone)
        results = []

        for _, row in recommendations.iterrows():
            description = row["description"]
            truncated = " ".join(description.split()[:30]) + "..."

            authors_split = row["authors"].split(";")
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = row["authors"]

            results.append({
                "title": row["title"],
                "authors": authors_str,
                "description": truncated,
                "large_thumbnail": row["large_thumbnail"],
                "info_link": row.get("info_link", "")
            })

        return {"recommendations": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
