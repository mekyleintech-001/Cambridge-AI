import streamlit as st, pickle, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Cambridge AI")
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def load_brain():
    chunks = pickle.load(open("chunks.pkl","rb"))
    index = faiss.read_index("model.faiss")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return chunks, index, model

chunks, index, embed_model = load_brain()
st.title("Cambridge AI - Ready")
st.write(f"Loaded {len(chunks)} chapters")

query = st.text_input("Ask anything...")

if query:
    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb).astype('float32'), 3)
    context = "\n\n".join([chunks[i] for i in I[0]])
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"system","content":f"Context: {context}"},
            {"role":"user","content":query}
        ],
        max_tokens=800
    )
    st.write(r.choices[0].message.content)
