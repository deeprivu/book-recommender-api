import os
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain (older style imports)
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
# If you migrate to the new modular layout later:
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OpenAIEmbeddings


# ------------------------------------------------------------------
# Environment & Config
# ------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️  WARNING: OPENAI_API_KEY not set. Embeddings will fail if invoked.")

PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")
BOOKS_CSV = "books_with_emotions.csv"
FALLBACK_THUMB = "./cover-not-found.jpg"

TOP_K_INITIAL = 50
TOP_K_FINAL = 16

# Tone → column mapping
TONE_COLUMN_MAP = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness"
}

# ------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------
app = FastAPI(title="Semantic Book Recommender API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Narrow this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------
if not os.path.exists(BOOKS_CSV):
    raise FileNotFoundError(f"Books dataset not found: {BOOKS_CSV}")

books = pd.read_csv(BOOKS_CSV)
print(f"✅ Books dataset loaded with {len(books)} entries.")

# Create large thumbnail column safely
books["thumbnail"] = books["thumbnail"].fillna("")
books["large_thumbnail"] = np.where(
    books["thumbnail"] == "",
    FALLBACK_THUMB,
    books["thumbnail"] + "&fife=w800"
)

# Normalize isbn13 column as string
if "isbn13" not in books.columns:
    raise ValueError("Dataset must contain 'isbn13' column.")
books["isbn13"] = books["isbn13"].astype(str).str.strip()

# ------------------------------------------------------------------
# Chroma Vector Store
# ------------------------------------------------------------------
if not os.path.exists(PERSIST_DIRECTORY):
    print(f"⚠️  Chroma directory '{PERSIST_DIRECTORY}' not found. "
          f"Ensure you have ingested before querying.")
db_books = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=OpenAIEmbeddings()
)

try:
    collection_count = db_books._collection.count()
except Exception:
    collection_count = "UNKNOWN"

print(f"🗄  Chroma DB loaded from: {PERSIST_DIRECTORY}")
print(f"📦 Collection count: {collection_count}")

if isinstance(collection_count, int) and collection_count == 0:
    print("⚠️  Vector store is empty; all queries will return zero results.")


# ------------------------------------------------------------------
# Request / Response Models
# ------------------------------------------------------------------
class RecommendationRequest(BaseModel):
    query: str
    category: str | None = None
    tone: str | None = None


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
ISBN_REGEX = re.compile(r"(97[89][0-9]{10})")

def extract_isbn(raw_page_content: str) -> str | None:
    """
    Extract a 13-digit ISBN (starting with 978 or 979) from a Chroma document's page_content.
    Returns the first match or None.
    """
    if not raw_page_content:
        return None
    m = ISBN_REGEX.search(raw_page_content)
    return m.group(1) if m else None


def truncate_text(text: str, max_words: int = 30) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def format_authors(authors_raw: str) -> str:
    if not authors_raw:
        return ""
    parts = [a.strip() for a in authors_raw.split(";") if a.strip()]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    if len(parts) > 2:
        return f"{', '.join(parts[:-1])}, and {parts[-1]}"
    return parts[0] if parts else ""


# ------------------------------------------------------------------
# Core Retrieval
# ------------------------------------------------------------------
def retrieve_semantic_recommendations(
        query: str,
        category: str | None = None,
        tone: str | None = None,
        initial_top_k: int = TOP_K_INITIAL,
        final_top_k: int = TOP_K_FINAL,
) -> pd.DataFrame:

    if not query or not query.strip():
        return pd.DataFrame()

    try:
        recs = db_books.similarity_search(query, k=initial_top_k)
    except Exception as e:
        print(f"❌ similarity_search error: {e}")
        return pd.DataFrame()

    if not recs:
        print("ℹ️  No vector matches returned.")
        return pd.DataFrame()

    # Extract clean ISBNs from page_content
    isbn_candidates = []
    for r in recs:
        isbn = extract_isbn(r.page_content)
        if isbn:
            isbn_candidates.append(isbn)

    if not isbn_candidates:
        print("ℹ️  No ISBNs extracted from vector results.")
        return pd.DataFrame()

    # Filter base set
    matched = books[books["isbn13"].isin(isbn_candidates)]

    if matched.empty:
        print("ℹ️  No rows in CSV matched extracted ISBNs.")
        return matched

    # Category filtering
    if category and category != "All":
        if "simple_categories" in matched.columns:
            matched = matched[matched["simple_categories"] == category]
        else:
            print("⚠️  'simple_categories' column missing; skipping category filter.")

    if matched.empty:
        return matched.head(final_top_k)

    # Tone sorting
    if tone and tone in TONE_COLUMN_MAP:
        tone_col = TONE_COLUMN_MAP[tone]
        if tone_col in matched.columns:
            matched = matched.sort_values(by=tone_col, ascending=False)
        else:
            print(f"⚠️  Tone column '{tone_col}' not found; skipping tone sort.")

    # Limit final size
    return matched.head(final_top_k)


# ------------------------------------------------------------------
# API Endpoint
# ------------------------------------------------------------------
@app.post("/recommend")
def recommend_books(req: RecommendationRequest):
    print(f"➡️  Received request: query='{req.query}' category='{req.category}' tone='{req.tone}'")
    try:
        df = retrieve_semantic_recommendations(req.query, req.category, req.tone)
        results = []

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            results.append({
                "_id": str(i),
                "title": row.get("title", "Unknown Title"),
                "authors": format_authors(row.get("authors", "")),
                "description": truncate_text(row.get("description", "") or ""),
                "large_thumbnail": row.get("large_thumbnail", FALLBACK_THUMB),
                "info_link": row.get("info_link", ""),
                "isbn13": row.get("isbn13", "")
            })

        print(f"✅ Returning {len(results)} recommendation(s).")
        return {"recommendations": results}

    except Exception as e:
        print(f"❌ Error in /recommend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Root Route
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "📚 Book Recommender API is running!", "collection_count": collection_count}
