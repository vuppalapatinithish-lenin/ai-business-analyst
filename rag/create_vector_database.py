import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CHROMA DATABASE
# ============================================================

client = chromadb.PersistentClient(
    path="vector_db"
)


# ============================================================
# READ COMPANY HANDBOOK
# ============================================================

with open(
    "documents/company_handbook.txt",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


# ============================================================
# CHUNK DOCUMENT
# ============================================================

chunks = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]


print(f"Total chunks: {len(chunks)}")


# ============================================================
# CREATE TF-IDF EMBEDDINGS
# ============================================================

vectorizer = TfidfVectorizer()

embeddings = vectorizer.fit_transform(
    chunks
).toarray()


# ============================================================
# DELETE OLD COLLECTION
# ============================================================

try:

    client.delete_collection(
        name="company_documents"
    )

    print("Old collection deleted.")

except Exception:

    pass


# ============================================================
# CREATE COLLECTION
# ============================================================

collection = client.create_collection(
    name="company_documents"
)


# ============================================================
# STORE DOCUMENTS + EMBEDDINGS
# ============================================================

collection.add(

    ids=[
        f"chunk_{i}"
        for i in range(len(chunks))
    ],

    documents=chunks,

    embeddings=embeddings.tolist()
)


print("Documents stored successfully! ✅")
print("Vector database created successfully! ✅")