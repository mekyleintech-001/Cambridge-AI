import streamlit as st
import pickle
import faiss
import numpy as np
from groq import Groq

st.set_page_config(page_title="Cambridge AI")
st.title("Cambridge AI - Loading...")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def load_brain():
    from sentence_transformers import SentenceTransformer
    chunks = pickle.load(open("chunks.pkl","rb"))
    index = faiss.read_index("model.faiss")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return chunks, index, model

chunks, index, embed_model = load_brain()
st.write(f"Ready: {len(chunks)} chapters loaded")

query = st.text_input("Ask anything...")

if query:
    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb).astype('float32'), 3)
    context = "\n\n".join([chunks[i] for i in I[0]])
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"system","content":f"Use this context: {context}"},
            {"role":"user","content":query}
        ],
        max_tokens=800
    )
    st.write(r.choices[0].message.content)
