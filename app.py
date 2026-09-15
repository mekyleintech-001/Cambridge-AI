import streamlit as st
import pickle, faiss, numpy as np
from groq import Groq

st.set_page_config(page_title="Cambridge AI", page_icon="🎓", layout="centered")

st.markdown("""
<style>
.stApp { background: #0e1117; }
h1 { color: #FFD700!important; font-weight:800; letter-spacing: -1px; }
.by { color:#8b949e; font-size:14px; margin-top:-15px; margin-bottom:25px; }
.chat-bubble { background:#1e222d; padding:18px 20px; border-radius:18px; margin:8px 0; line-height:1.6; }
.user-bubble { background:#2a3f5f; border-left:3px solid #4da6ff; }
.ai-bubble { border-left:3px solid #FFD700; }
.typing { display:flex; gap:6px; padding:12px 0; }
.dot { width:10px; height:10px; background:#FFD700; border-radius:50%; animation: bounce 1.4s infinite; }
.dot:nth-child(2){ animation-delay:.2s }.dot:nth-child(3){ animation-delay:.4s }
@keyframes bounce { 0%,60%,100%{ transform:translateY(0); opacity:.4 } 30%{ transform:translateY(-8px); opacity:1 } }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎓 Cambridge AI")
    st.markdown("by Keane Moyo")
    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
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

st.title("Cambridge AI")
st.markdown('<div class="by">by Keane Moyo</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"Hey! Ask me anything 📚"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        css = "user-bubble" if m["role"]=="user" else "ai-bubble"
        st.markdown(f'<div class="chat-bubble {css}">{m["content"]}</div>', unsafe_allow_html=True)

if query := st.chat_input("ask about anything..."):
    st.session_state.messages.append({"role":"user","content":query})
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-bubble user-bubble">{query}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        bubble_placeholder = st.empty()
        bubble_placeholder.markdown('<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>', unsafe_allow_html=True)

        q_emb = embed_model.encode([query])
        D, I = index.search(np.array(q_emb).astype('float32'), 3)
        context = "\n\n".join([chunks[i] for i in I[0]])

        r = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role":"system","content": f"You are Cambridge AI by Keane Moyo. Answer directly based on the user's prompt and the provided context. Stay strictly on what they asked. Do not bring unrelated topics. Be helpful and complete in answering their specific question. Context: {context}"},
                {"role":"user","content": query}
            ],
            max_tokens=1000,
            temperature=0.4
        )
        ans = r.choices[0].message.content
        bubble_placeholder.empty()
        st.markdown(f'<div class="chat-bubble ai-bubble">{ans}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role":"assistant","content":ans})
