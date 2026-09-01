import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------
# 1. Connect to Vector Database
# --------------------------------

chroma_client = chromadb.PersistentClient(
    path="vector_db"
)

collection = chroma_client.get_collection(
    name="company_documents"
)


# --------------------------------
# 2. Get stored documents
# --------------------------------

stored_data = collection.get(
    include=["documents"]
)

documents = stored_data["documents"]


# --------------------------------
# 3. Create vectorizer
# --------------------------------

vectorizer = TfidfVectorizer()

document_vectors = vectorizer.fit_transform(
    documents
)


# --------------------------------
# 4. Ask question
# --------------------------------

question = input("\nAsk a question: ")


# --------------------------------
# 5. Convert question to vector
# --------------------------------

question_vector = vectorizer.transform(
    [question]
)


# --------------------------------
# 6. Find relevant document
# --------------------------------

result = collection.query(
    query_embeddings=question_vector.toarray().tolist(),
    n_results=1
)


# --------------------------------
# 7. Retrieve context
# --------------------------------

context = result["documents"][0][0]


# --------------------------------
# 8. Display RAG context
# --------------------------------

print("\n--- RETRIEVED CONTEXT ---")
print(context)


# --------------------------------
# 9. Simple answer generation
# --------------------------------

print("\n--- AI ANSWER ---")

if "annual leave" in context.lower():

    print(
        "Employees receive 18 annual leave days per year."
    )

elif "work from home" in context.lower():

    print(
        "Employees can work from home up to two days "
        "per week with manager approval."
    )

elif "working hours" in context.lower():

    print(
        "Standard working hours are 9:00 AM to 6:00 PM, "
        "Monday to Friday."
    )

elif "performance" in context.lower():

    print(
        "Employee performance is reviewed twice every year."
    )

elif "training" in context.lower():

    print(
        "New employees receive an onboarding program "
        "during their first month."
    )

else:

    print(
        "I found relevant information, but I could not "
        "generate a specific answer."
    )