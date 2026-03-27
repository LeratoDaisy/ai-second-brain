# query.py
import os
import numpy as np
import faiss
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load your vectorstore embeddings from a file
def load_embeddings(file_path="vectorstore/chunks.npy"):
    if not os.path.exists(file_path):
        return None, None
    data = np.load(file_path, allow_pickle=True)
    embeddings = np.array([item[0] for item in data])
    texts = [item[1] for item in data]
    return embeddings, texts

# Function that app.py calls
def ask_question(question):
    embeddings, texts = load_embeddings()
    if embeddings is None:
        return "No PDF processed yet."
    
    # Compute embedding for the question
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    question_vector = np.array(resp.data[0].embedding).astype("float32")

    # FAISS similarity search
    index = faiss.IndexFlatL2(len(question_vector))
    index.add(embeddings)
    D, I = index.search(np.array([question_vector]), k=1)

    answer_text = texts[I[0][0]]
    
    # Call OpenAI GPT to generate an answer
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer this question based on the following context: {answer_text}\n\nQuestion: {question}"}
        ]
    )
    return completion.choices[0].message.content