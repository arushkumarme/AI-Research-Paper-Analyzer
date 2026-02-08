from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_embedding_model():
    model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return model


def create_vector_store(chunks):
    embedding_model = create_embedding_model()

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embedding_model
    )

    return vector_store


def retrieve_documents(vector_store, query):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(query)
    return docs




def keyword_search(chunks, query, top_k=3):
    results = []

    query_words = query.lower().split()

    for chunk in chunks:
        chunk_lower = chunk.lower()
        
        score = sum(word in chunk_lower for word in query_words)

        if score > 0:
            results.append((score, chunk))

    # Sort by keyword match score
    results.sort(reverse=True, key=lambda x: x[0])

    return [chunk for _, chunk in results[:top_k]]







def hybrid_search(vector_store, chunks, query):
    
    # Semantic Search
    semantic_docs = retrieve_documents(vector_store, query)
    semantic_chunks = [doc.page_content for doc in semantic_docs]

    # Keyword Search
    keyword_chunks = keyword_search(chunks, query)

    # Combine + Remove duplicates
    combined = list(set(semantic_chunks + keyword_chunks))

    return combined
