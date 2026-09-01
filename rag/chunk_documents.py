import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# --------------------------------
# 1. Read the company handbook
# --------------------------------

with open(
    "documents/company_handbook.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# --------------------------------
# 2. Split document into sections
# --------------------------------

section_pattern = r"""
(?=
LEAVE POLICY
|WORK FROM HOME POLICY
|WORKING HOURS
|PERFORMANCE POLICY
|TRAINING POLICY
)
"""

chunks = re.split(
    section_pattern,
    text,
    flags=re.VERBOSE
)

chunks = [
    chunk.strip()
    for chunk in chunks
    if chunk.strip()
]


# --------------------------------
# 3. Create TF-IDF vectors
# --------------------------------

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(chunks)

embeddings = vectors.toarray().tolist()


# --------------------------------
# 4. Create ChromaDB
# --------------------------------

chroma_client = chromadb.PersistentClient(
    path="vector_db"
)


# --------------------------------
# 5. Create collection
# --------------------------------

collection = chroma_client.get_or_create_collection(
    name="company_documents"
)


# --------------------------------
# 6. Clear old records
# --------------------------------

existing = collection.get()

if existing["ids"]:
    collection.delete(
        ids=existing["ids"]
    )


# --------------------------------
# 7. Store all chunks
# --------------------------------

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)


# --------------------------------
# 8. Show result
# --------------------------------

print("Vector Database updated successfully! ✅")
print("Total chunks stored:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\nCHUNK {i}")
    print("--------------------")
    print(chunk)