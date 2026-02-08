print("Starting vector test...")

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager

processor = DocumentProcessor()
vector_store = VectorStoreManager()

doc = processor.process_document("sample_papers/test.pdf")
vector_store.add_documents(doc["chunks"])

results = vector_store.search("What is transformer?")
print(results["documents"][0])
