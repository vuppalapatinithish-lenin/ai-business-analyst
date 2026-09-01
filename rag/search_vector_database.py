import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONNECT TO CHROMA
# ============================================================

client = chromadb.PersistentClient(
    path="vector_db"
)

collection = client.get_collection(
    name="company_documents"
)


# ============================================================
# GET ALL STORED DOCUMENTS
# ============================================================

data = collection.get(
    include=["documents", "embeddings"]
)

documents = data["documents"]


# ============================================================
# REBUILD TF-IDF MODEL
# ============================================================

vectorizer = TfidfVectorizer()

document_embeddings = vectorizer.fit_transform(
    documents
).toarray()


# ============================================================
# ASK QUESTION
# ============================================================

question = input(
    "Ask a question: "
)


# ============================================================
# CREATE QUESTION EMBEDDING
# ============================================================

question_embedding = vectorizer.transform(
    [question]
).toarray()[0]


# ============================================================
# SEARCH CHROMA
# ============================================================

results = collection.query(
    query_embeddings=[
        question_embedding.tolist()
    ],
    n_results=3
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n--- RETRIEVED DOCUMENTS ---")

for i, document in enumerate(
    results["documents"][0],
    start=1
):

    print(f"\nRESULT {i}")
    print("-------------------------")
    print(document)