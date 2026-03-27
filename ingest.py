import os
from PyPDF2 import PdfReader
import faiss
import numpy as np
from openai import OpenAI

# Read API key from environment
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Make sure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

def process_file(path):
    # Read PDF
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # avoid None
            text += page_text

    if not text:
        raise ValueError("PDF has no extractable text")

    # Chunk text into 500-character pieces
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # Create embeddings
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)

    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, "vectorstore/index.faiss")

    # Save chunks
    with open("vectorstore/chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n---\n")

    print(f"Processed {len(chunks)} chunks from {path}")