import streamlit as st
import pickle, faiss, numpy as np, uuid, datetime
from groq import Groq

st.set_page_config(page_title="Cambridge AI", page_icon="🎓", layout="wide")

# --- CSS: GLOW + STYLISH BUTTONS ---
st.markdown("""
<style>
.stApp { background: #0e1117; }
h1 { color: #FFD700!important; font-weight:800; }
.by { color:#8b949e; font-size:14px; margin-top:-15px; margin-bottom:25px; }
.chat-bubble { background:#1e222d; padding:18px 20px; border-radius:18px; margin:8px 0; line-height:1.6; position:relative; }
.user-bubble { background:#2a3f5f; border-left:3px solid #4da6ff; }
.ai-bubble { border-left:3px solid #FFD700; }

/* Floating dots */
.typing { display:flex; gap:6px; padding:12px 0; }
.dot { width:10px; height:10px; background:#FFD700; border-radius:50%; animation: bounce 1.4s infinite; }
.dot:nth-child(2){ animation-delay:.2s }.dot:nth-child(3){ animation-delay:.4s }
@keyframes bounce { 0%,60%,100%{ transform:translateY(0); opacity:.4 } 30%{ transform:translateY(-8px); opacity:1 } }

/* BLUE GLOW EFFECT */
div[data-testid="stChatInput"] > div {
    border:1px solid #333!important;
    background:#1e222d!important;
    transition: all 0.4s ease!important;
}
div[data-testid="stChatInput"]:hover > div {
    border-color:#4da6ff!important;
    box-shadow: 0 0 20px rgba(77,166,255,0.5), 0 0 40px rgba(77,166,255,0.2)!important;
    transform: translateY(-1px);
}
div[data-testid="stChatInput"]:focus-within > div {
    border-color:#4da6ff!important;
    box-shadow: 0 0 25px rgba(77,166,255,0.8), 0 0 50px rgba(77,166,255,0.4)!important;
}

/* Chat list buttons */
.chat-item { 
    background:#1e222d; border:1px solid #2a2f3a; padding:10px 12px; 
    border-radius:10px; margin:5px 0; cursor:pointer; font-size:13px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.chat-item:hover { border-color:#4da6ff; background:#2a3f5f; }
.chat-active { border-color:#FFD700!important; background:#2a2a1e!important; }

/* Action icons row */
.actions { display:flex; gap:8px; margin-top:8px; }
.icon-btn { 
    background:#1e222d; border:1px solid #333; border-radius:20px; 
    padding:4px 10px; font-size:11px; color:#8b949e; cursor:pointer;
    transition: all 0.2s;
}
.icon-btn:hover { color:white; border-color:#4da6ff; background:#2a3f5f; transform:scale(1.05); }
</style>
""", unsafe_allow_html=True)

# --- INIT CHATS SYSTEM ---
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    new_id = str(uuid.uuid4())[:8]
    st.session_state.current_chat = new_id
    st.session_state.chats[new_id] = {"title":"New Chat","time":datetime.datetime.now().strftime("%H:%M"), "messages":[{"role":"assistant","content":"Hey! Ask me anything 📚"}]}

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def load_brain():
    from sentence_transformers import SentenceTransformer
    chunks = pickle.load(open("chunks.pkl","rb"))
    index = faiss.read_index("model.faiss")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return chunks, index, model

chunks, index, embed_model = load_brain()

# --- SIDEBAR WITH OLD CHATS ---
with st.sidebar:
    st.markdown("## 🎓 Cambridge AI")
    st.markdown("by Keane Moyo")
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.current_chat = new_id
        st.session_state.chats[new_id] = {"title":"New Chat","time":datetime.datetime.now().strftime("%H:%M"), "messages":[{"role":"assistant","content":"Hey! Ask me anything 📚"}]}
        st.rerun()

    st.markdown("### 🕓 Old Chats")
    # Show chats newest first
    for chat_id, chat_data in reversed(list(st.session_state.chats.items())):
        is_active = chat_id == st.session_state.current_chat
        title = chat_data["title"][:30]
        if st.button(f"{'🟡 ' if is_active else '💬 '}{title}", key=f"chat_{chat_id}", use_container_width=True):
            st.session_state.current_chat = chat_id
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Delete Current Chat", use_container_width=True):
        if len(st.session_state.chats) > 1:
            del st.session_state.chats[st.session_state.current_chat]
            st.session_state.current_chat = list(st.session_state.chats.keys())[-1]
            st.rerun()
        else:
            st.toast("Can't delete last chat")

# --- MAIN AREA ---
current = st.session_state.chats[st.session_state.current_chat]
st.title("Cambridge AI")
st.markdown('<div class="by">by Keane Moyo</div>', unsafe_allow_html=True)

# Display messages with stylish copy
for idx, m in enumerate(current["messages"]):
    with st.chat_message(m["role"]):
        css = "user-bubble" if m["role"]=="user" else "ai-bubble"
        st.markdown(f'<div class="chat-bubble {css}">{m["content"]}</div>', unsafe_allow_html=True)
        
        # STYLISH copy/delete row - no replicate
        st.markdown(f"""
        <div class="actions">
            <button class="icon-btn" onclick="navigator.clipboard.writeText(`{m['content'].replace('`','').replace(chr(10),' ' )[:2000]}`); this.innerText='✅ Copied'">📋 Copy</button>
            <span style="font-size:11px;color:#555;">{current['time']}</span>
        </div>
        """, unsafe_allow_html=True)

# Input
if query := st.chat_input("ask about anything..."):
    # Update title from first message
    if current["title"]=="New Chat":
        current["title"] = query[:35]
    current["messages"].append({"role":"user","content":query})
    
    # Generate answer
    with st.chat_message("assistant"):
        bubble = st.empty()
        bubble.markdown('<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>', unsafe_allow_html=True)

        q_emb = embed_model.encode([query])
        D, I = index.search(np.array(q_emb).astype('float32'), 3)
        context = "\n\n".join([chunks[i] for i in I[0]])

        r = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role":"system","content": f"Answer based on user's prompt. Context: {context}"},
                {"role":"user","content": query}
            ],
            max_tokens=1000
        )
        ans = r.choices[0].message.content
        bubble.empty()
        st.markdown(f'<div class="chat-bubble ai-bubble">{ans}</div>', unsafe_allow_html=True)
        current["messages"].append({"role":"assistant","content":ans})
        st.rerun()
