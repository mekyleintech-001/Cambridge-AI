import streamlit as st, pickle, faiss, numpy as np, uuid, datetime
from groq import Groq

st.set_page_config(page_title="Cambridge AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at top, #1a2235 0%, #0e1117 70%); }
h1 { color: #FFD700!important; font-weight:800; letter-spacing:-1px; }
.by { color:#8b949e; font-size:14px; margin-top:-15px; margin-bottom:25px; }

/* LIQUID GLASS BUBBLES */
.chat-bubble {
    background: rgba(30, 34, 45, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    padding:18px 22px; border-radius:20px; margin:10px 0; line-height:1.65;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
    animation: slideUp 0.4s ease-out;
    transition: all 0.3s ease;
}
.chat-bubble:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,0,0,0.4); border-color: rgba(255,255,255,0.15); }
.user-bubble {
    background: linear-gradient(135deg, rgba(42,63,95,0.8), rgba(32,50,80,0.6));
    border-left:3px solid #4da6ff; border:1px solid rgba(77,166,255,0.2);
}
.ai-bubble {
    background: linear-gradient(135deg, rgba(30,34,45,0.7), rgba(25,28,38,0.6));
    border-left:3px solid #FFD700;
}
@keyframes slideUp { from { opacity:0; transform: translateY(15px); } to { opacity:1; transform: translateY(0); } }

/* TYPING DOTS - smooth */
.typing { display:flex; gap:7px; padding:14px; }
.dot { width:10px; height:10px; background:linear-gradient(135deg,#FFD700,#ffaa00); border-radius:50%; animation: bounce 1.4s infinite ease-in-out; box-shadow:0 0 10px rgba(255,215,0,0.5); }
.dot:nth-child(2){ animation-delay:.2s }.dot:nth-child(3){ animation-delay:.4s }
@keyframes bounce { 0%,80%,100%{ transform:scale(0.8); opacity:0.5 } 40%{ transform:scale(1.2); opacity:1 } }

/* TEXTBOX - GLOW ON HOVER, BLUE EDGE ON CLICK */
div[data-testid="stChatInput"] > div {
    background: rgba(30,34,45,0.6)!important;
    backdrop-filter: blur(12px)!important;
    border:1px solid rgba(255,255,255,0.1)!important;
    border-radius:16px!important;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1)!important;
}
div[data-testid="stChatInput"]:hover > div {
    border-color: rgba(77,166,255,0.6)!important;
    box-shadow: 0 0 20px rgba(77,166,255,0.4), 0 0 40px rgba(77,166,255,0.15)!important;
    transform: translateY(-1px);
}
div[data-testid="stChatInput"]:focus-within > div {
    border:2px solid #4da6ff!important;
    box-shadow: none!important;
    background: rgba(30,34,45,0.8)!important;
}

/* LEVEL PILLS */
.level-pill {
    padding:6px 14px; border-radius:20px; font-size:12px; font-weight:600;
    border:1px solid rgba(255,255,255,0.1); background:rgba(30,34,45,0.5);
    backdrop-filter: blur(8px); cursor:pointer; transition: all 0.3s; color:#8b949e;
}
.level-active { background: linear-gradient(135deg, #4da6ff, #2a7de1)!important; color:white!important; border-color:#4da6ff!important; box-shadow:0 0 15px rgba(77,166,255,0.4); }
</style>
""", unsafe_allow_html=True)

if "chats" not in st.session_state: st.session_state.chats = {}
if "current_chat" not in st.session_state:
    nid=str(uuid.uuid4())[:8]
    st.session_state.current_chat=nid
    st.session_state.chats[nid]={"title":"New Chat","time":datetime.datetime.now().strftime("%H:%M"),"messages":[{"role":"assistant","content":"Hey! Ask me anything 📚"}]}
if "level" not in st.session_state: st.session_state.level="Simple"

client=Groq(api_key=st.secrets["GROQ_API_KEY"])
@st.cache_resource
def load_brain():
    from sentence_transformers import SentenceTransformer
    chunks=pickle.load(open("chunks.pkl","rb"))
    index=faiss.read_index("model.faiss")
    model=SentenceTransformer('all-MiniLM-L6-v2')
    return chunks,index,model
chunks,index,embed_model=load_brain()

# SIDEBAR
with st.sidebar:
    st.markdown("## 🎓 Cambridge AI")
    st.markdown("by Keane Moyo")
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        nid=str(uuid.uuid4())[:8]; st.session_state.current_chat=nid
        st.session_state.chats[nid]={"title":"New Chat","time":datetime.datetime.now().strftime("%H:%M"),"messages":[{"role":"assistant","content":"Hey! Ask me anything 📚"}]}
        st.rerun()
    st.markdown("### 🕓 Old Chats")
    for cid,cdata in reversed(list(st.session_state.chats.items())):
        if st.button(f"{'🟡 ' if cid==st.session_state.current_chat else '💬 '}{cdata['title'][:28]}", key=f"chat_{cid}", use_container_width=True):
            st.session_state.current_chat=cid; st.rerun()
    st.markdown("---")
    if st.button("🗑️ Delete Current", use_container_width=True):
        if len(st.session_state.chats)>1:
            del st.session_state.chats[st.session_state.current_chat]
            st.session_state.current_chat=list(st.session_state.chats.keys())[-1]; st.rerun()

current=st.session_state.chats[st.session_state.current_chat]
st.title("Cambridge AI")
st.markdown('<div class="by">by Keane Moyo</div>', unsafe_allow_html=True)

# MESSAGES
for m in current["messages"]:
    with st.chat_message(m["role"]):
        css="user-bubble" if m["role"]=="user" else "ai-bubble"
        st.markdown(f'<div class="chat-bubble {css}">{m["content"]}</div>', unsafe_allow_html=True)
        st.markdown(f"<button class='level-pill' onclick=\"navigator.clipboard.writeText(`{m['content'].replace('`','').replace(chr(10),' ')[:3000]}`)\">📋 Copy</button>", unsafe_allow_html=True)

# --- LEVEL SELECTOR RIGHT ON TEXTBOX AREA ---
col1,col2,col3,spacer = st.columns([1,1,1,3])
with col1:
    if st.button("Simple", key="lvl_simple", help="400 tokens - quick"):
        st.session_state.level="Simple"; st.rerun()
with col2:
    if st.button("Moderate", key="lvl_mod", help="750 tokens"):
        st.session_state.level="Moderate"; st.rerun()
with col3:
    if st.button("Best", key="lvl_best", help="1000 tokens - detailed"):
        st.session_state.level="Best"; st.rerun()

# Show active level
token_map={"Simple":400,"Moderate":750,"Best":1000}
active=st.session_state.level
st.markdown(f"""
<div style="margin:8px 0 12px 0; display:flex; gap:8px; align-items:center;">
    <span style="font-size:12px; color:#8b949e;">Level:</span>
    <span class="level-pill {'level-active' if active=='Simple' else ''}">Simple • 400</span>
    <span class="level-pill {'level-active' if active=='Moderate' else ''}">Moderate • 750</span>
    <span class="level-pill {'level-active' if active=='Best' else ''}">Best • 1000</span>
</div>
""", unsafe_allow_html=True)

# INPUT
if query := st.chat_input("ask about anything..."):
    if current["title"]=="New Chat": current["title"]=query[:35]
    current["messages"].append({"role":"user","content":query})
    with st.chat_message("assistant"):
        b=st.empty(); b.markdown('<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>', unsafe_allow_html=True)
        q_emb=embed_model.encode([query])
        D,I=index.search(np.array(q_emb).astype('float32'),3)
        context="\n\n".join([chunks[i] for i in I[0]])
        r=client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role":"system","content":f"Answer based on prompt. Context:{context}"},{"role":"user","content":query}],
            max_tokens=token_map[st.session_state.level]
        )
        ans=r.choices[0].message.content; b.empty()
        st.markdown(f'<div class="chat-bubble ai-bubble">{ans}</div>', unsafe_allow_html=True)
        current["messages"].append({"role":"assistant","content":ans})
        st.rerun()
