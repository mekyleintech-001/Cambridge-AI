import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Cambridge AI", page_icon="📚")

# Load secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
WORKING_MODEL = "llama-3.3-70b-versatile"

# Load your brain files
@st.cache_resource
def load_brain():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("model.faiss")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return chunks, index, model

chunks, index, embed_model = load_brain()

st.title("Cambridge AI - Ready")
st.write(f"Loaded {len(chunks)} chapters")

query = st.text_input("Ask anything...")

if query:
    # Search
    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb).astype('float32'), 3)
    context = "\n\n".join([chunks[i] for i in I[0]])

    # Ask Groq
    r = client.chat.completions.create(
        model=WORKING_MODEL,
        messages=[
            {"role": "system", "content": f"You are a Cambridge exam tutor. Use this context:\n{context}"},
            {"role": "user", "content": query}
        ],
        max_tokens=800
    )
    st.write(r.choices[0].message.content)
