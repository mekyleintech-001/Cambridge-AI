import streamlit as st
import pickle, faiss, numpy as np
from groq import Groq

st.set_page_config(page_title="Cambridge AI", page_icon="🎓", layout="wide")

# --- COOL CSS ---
st.markdown("""
<style>
.stApp { background: #0e1117; }
h1 { color: #FFD700!important; font-weight: 800; }
.chat-bubble { background: #1e222d; padding: 18px; border-radius: 16px; border-left: 4px solid #FFD700; margin: 10px 0; }
.user-bubble { background: #2a3f5f; border-left: 4px solid #4da6ff; }
.small { color: #8b949e; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🎓 Cambridge AI")
    st.markdown("---")
    st.success("✅ 483 chapters loaded")
    st.markdown("### How it works")
    st.markdown("I search your 483 Cambridge textbooks and answer with Groq AI.")
    st.markdown("---")
    st.caption("Powered by Groq • gpt-oss-20b • FAISS")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def load_brain():
    from sentence_transformers import SentenceTransformer
    chunks = pickle.load(open("chunks.pkl","rb"))
    index = faiss.read_index("model.faiss")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return chunks, index, model

chunks, index, embed_model = load_brain()

# --- HEADER ---
col1, col2 = st.columns([3,1])
with col1:
    st.title("Cambridge AI")
    st.markdown('<p class="small">Your personal tutor for 483 chapters • Ask anything</p>', unsafe_allow_html=True)
with col2:
    st.metric("Chapters", len(chunks))

# --- CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"Hi! I'm your Cambridge tutor. Ask me anything from your 483 chapters 📚"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(f'<div class="chat-bubble {"user-bubble" if m["role"]=="user" else ""}">{m["content"]}</div>', unsafe_allow_html=True)

# --- INPUT ---
if query := st.chat_input("Ask about Unicode, Python, Cambridge..."):
    st.session_state.messages.append({"role":"user","content":query})
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-bubble user-bubble">{query}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("Searching 483 chapters..."):
            q_emb = embed_model.encode([query])
            D, I = index.search(np.array(q_emb).astype('float32'), 3)
            context = "\n\n".join([chunks[i] for i in I[0]])

            r = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role":"system","content":f"You are a friendly Cambridge exam tutor. Explain clearly with bullet points. Context:\n{context}"},
                    {"role":"user","content":query}
                ],
                max_tokens=800
            )
            ans = r.choices[0].message.content

        st.markdown(f'<div class="chat-bubble">{ans}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role":"assistant","content":ans})
