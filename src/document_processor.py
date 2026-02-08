import fitz
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter



# -----------------------------
# Load PDF and extract text
# -----------------------------
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text


# -----------------------------
# Extract metadata
# -----------------------------
def extract_metadata(file_path):
    doc = fitz.open(file_path)
    return doc.metadata


# -----------------------------
# Split text by sections
# -----------------------------
def split_by_sections(text):
    pattern = r"\n[A-Z][A-Za-z\s]{3,}\n"
    sections = re.split(pattern, text)
    return sections


# -----------------------------
# Chunk text for AI
# -----------------------------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)


# -----------------------------
# Extract layout blocks (tables, figures etc.)
# -----------------------------
def extract_blocks(file_path):
    doc = fitz.open(file_path)
    blocks_data = []

    for page in doc:
        blocks = page.get_text("blocks")
        blocks_data.extend(blocks)

    return blocks_data


# -----------------------------
# Main processing pipeline
# -----------------------------
def process_document(file_path):
    text = load_pdf(file_path)
    metadata = extract_metadata(file_path)
    sections = split_by_sections(text)
    chunks = chunk_text(text)
    blocks = extract_blocks(file_path)

    return {
        "metadata": metadata,
        "sections": sections,
        "chunks": chunks,
        "blocks": blocks
    }






from vector_store import create_vector_store, retrieve_documents
from llm_agent import generate_answer


if __name__ == "__main__":
    file_path = "sample_papers/test.pdf"

    result = process_document(file_path)

    vector_store = create_vector_store(result["chunks"])

    while True:
        query = input("\nAsk your question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        docs = retrieve_documents(vector_store, query)
        retrieved_chunks = [doc.page_content for doc in docs]

        answer = generate_answer(query, retrieved_chunks)

        print("\nAI Assistant:\n")
        print(answer)

