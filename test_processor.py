from src.document_processor import DocumentProcessor

processor = DocumentProcessor()

result = processor.process_document("sample_papers/test.pdf")

print("Chunks created:", len(result["chunks"]))
print("\nSample chunk:\n")
print(result["chunks"][0]["content"][:300])
