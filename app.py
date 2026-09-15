import os, faiss, pickle, numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="CambridgeAI 9618", page_icon="🎓")
st.title("🎓 CambridgeAI 9618")

API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not API_KEY:
    st.error("Add GROQ_API_KEY in Streamlit Secrets")
    st.stop()

client = Groq(api_key=API_KEY)
WORKING_MODEL = "llama-3.3-70b-versatile"

@st.cache_resource
def load_data():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("model.faiss")
    chunks = pickle.load(open("chunks.pkl","rb"))
    return model, index, chunks

embed_model, index, chunks = load_data()
st.success(f"Ready: {len(chunks)} chapters | {WORKING_MODEL}")

style = st.radio("Style", ["Explain","Exam Question","Mark My Answer"], horizontal=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"):
        st.markdown(question)

    q_emb = embed_model.encode([question])
    D,I = index.search(np.array(q_emb), k=5)
    context = "\n".join([chunks[i] for i in I[0]])[:4000]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            r = client.chat.completions.create(
                model=WORKING_MODEL,
                messages=[
                    {"role":"system","content":f"Cambridge 9618 tutor, style:{style}. Context:{context}"},
                    {"role":"user","content":question}
                ],
                max_tokens=800
            )
            answer = r.choices[0].message.content
            st.markdown(answer)

    st.session_state.messages.append({"role":"assistant","content":answer})
