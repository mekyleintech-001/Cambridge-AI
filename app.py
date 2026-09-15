import streamlit as st, pickle, faiss, numpy as np, uuid, datetime, re
from groq import Groq

st.set_page_config(page_title="Cambridge AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at top, #1a2235 0%, #0e1117 70%); }
h1 { color: #FFD700!important; font-weight:800; }
.by { color:#8b949e; font-size:14px; margin-top:-15px; margin-bottom:20px; }
.chat-bubble {
    background: rgba(30, 34, 45, 0.65); backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.08); padding:16px 20px;
    border-radius:20px; margin:8px 0; animation: slideUp 0.35s ease;
}
@keyframes slideUp { from {opacity:0; transform:translateY(12px);} to {opacity:1; transform:translateY(0);} }
.user-bubble { background: linear-gradient(135deg, rgba(42,63,95,0.85), rgba(32,50,80,0.65)); }
.ai-bubble { background: rgba(30,34,45,0.6); }
.typing { display:flex; gap:6px; padding:12px; }
.dot { width:9px; height:9px; background:#FFD700; border-radius:50%; animation: bounce 1.4s infinite; }
@keyframes bounce { 0%,80%,100%{transform:scale(0.7);opacity:0.5} 40%{transform:scale(1.2);opacity:1} }

/* TEXTBOX + BUTTONS INSIDE */
div[data-testid="stChatInput"] {
    position: relative!important;
}
div[data-testid="stChatInput"] > div {
    background: rgba(30,34,45,0.7)!important; backdrop-filter: blur(12px)!important;
    border:1px solid rgba(255,255,255,0.1)!important; border-radius:24px!important;
    padding-right: 95px!important; /* make space for 2 icons */
}
div[data-testid="stChatInput"]:hover > div {
    border-color: rgba(77,166,255,0.5)!important; box-shadow: 0 0 18px rgba(77,166,255,0.35)!important;
}
div[data-testid="stChatInput"]:focus-within > div {
    border:2px solid #4da6ff!important; box-shadow: none!important;
}

/* 3 DOTS BUTTON NEXT TO SEND */
div[data-testid="stPopover"] {
    position: absolute!important;
    right: 48px!important; /* 48px = just left of send button */
    bottom: 6px!important;
    z-index: 9999!important;
}
div[data-testid="stPopover"] > button {
    background: rgba(255,255,255,0.08)!important;
    border:1px solid rgba(255,255,255,0.12)!important;
    border-radius:50%!important; width:32px!important; height:32px!important;
    color:#8b949e!important;
}
div[data-testid="stPopover"] > button:hover {
    background: rgba(77,166,255,0.2)!important; border-color:#4da6ff!important; color:white!important;
}
</style>
""", unsafe_allow_html=True)

if "chats" not in st.session_state: st.session_state.chats={}
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

def smart_finish(text, was_cut, level):
    text=text.strip()
    if not was_cut: return text
    lines=text.split('\n')
    junk=[r'^\s*Examples?\s*:?\s*$', r'^\s*\d+[\.\)]\s*$', r'^\s*\d+[\.\)]\s*\w{0,3}\s*$', r'^\s*[-•*]\s*$', r'^\s*Note\s*:?\s*$']
    while lines:
        last=lines[-1].strip()
        is_junk=any(re.match(p,last,re.IGNORECASE) for p in junk)
        is_short=len(last)<20 and not last.endswith('.') and not last.endswith('|') and not last.endswith('`')
        if is_junk or (is_short and len(lines)>2):
            lines.pop()
        else: break
    text='\n'.join(lines).strip()
    last_good=max(text.rfind('. '), text.rfind('.\n'), text.rfind('|\n'))
    if last_good>len(text)*0.5 and last_good!=-1:
        text=text[:last_good+1]
    text=re.sub(r'\n\s*Examples?\s*:?\s*$','',text,flags=re.IGNORECASE)
    text=re.sub(r'\n\s*\d+[\.\)]\s*$','',text)
    text=re.sub(r'\s*\d+[\.\)]\s*$','',text)
    if level=="Simple": text+="\n\n> 💡 Core covered. Switch to Moderate or Best for more."
    elif level=="Moderate": text+="\n\n> 💡 Want full table? Switch to Best."
    return text.strip()

with st.sidebar:
    st.markdown("## 🎓 Cambridge AI")
    st.caption("by Keane Moyo")
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        nid=str(uuid.uuid4())[:8]; st.session_state.current_chat=nid
        st.session_state.chats[nid]={"title":"New Chat","time":datetime.datetime.now().strftime("%H:%M"),"messages":[{"role":"assistant","content":"Hey! Ask me anything 📚"}]}
        st.rerun()
    st.markdown("### 🕓 Old Chats")
    for cid,cdata in reversed(list(st.session_state.chats.items())):
        if st.button(f"{'🟡 ' if cid==st.session_state.current_chat else '💬 '}{cdata['title'][:26]}", key=f"chat_{cid}", use_container_width=True):
            st.session_state.current_chat=cid; st.rerun()

current=st.session_state.chats[st.session_state.current_chat]
st.title("Cambridge AI")
st.markdown(f'<div class="by">by Keane Moyo • <span style="color:#4da6ff;">{st.session_state.level}</span></div>', unsafe_allow_html=True)

for m in current["messages"]:
    with st.chat_message(m["role"]):
        css="user-bubble" if m["role"]=="user" else "ai-bubble"
        st.markdown(f'<div class="chat-bubble {css}">{m["content"]}</div>', unsafe_allow_html=True)

# --- INPUT WITH 3 DOTS NEXT TO SEND ICON ---
# Popover must be declared BEFORE chat_input so CSS can move it
with st.popover("⋮", help="Change level"):
    st.markdown("**Answer Level**")
    lvl = st.radio("", ["Simple","Moderate","Best"], index=["Simple","Moderate","Best"].index(st.session_state.level), label_visibility="collapsed")
    if lvl!=st.session_state.level:
        st.session_state.level=lvl; st.rerun()
    st.caption("Simple: detailed + examples | Moderate: + depth | Best: full")

query = st.chat_input("ask about anything...")

if query:
    if current["title"]=="New Chat": current["title"]=query[:35]
    current["messages"].append({"role":"user","content":query})
    token_map={"Simple":400,"Moderate":750,"Best":1000}
    level_prompts={
        "Simple":"Explain in detail with clear examples. Must be complete within ~350 tokens. Do NOT start new sections like 'Examples' if you can't finish them. Be thorough but concise.",
        "Moderate":"Give Simple + add deeper explanation, comparisons. Keep complete within ~700 tokens.",
        "Best":"Full comprehensive answer with definition, examples, comparison table, advantages. Complete within 1000 tokens."
    }
    with st.chat_message("assistant"):
        b=st.empty(); b.markdown('<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>', unsafe_allow_html=True)
        q_emb=embed_model.encode([query]); D,I=index.search(np.array(q_emb).astype('float32'),3)
        context="\n\n".join([chunks[i] for i in I[0]])
        r=client.chat.completions.create(model="openai/gpt-oss-20b", messages=[{"role":"system","content":f"You are Cambridge AI. {level_prompts[st.session_state.level]} Context:{context}"},{"role":"user","content":query}], max_tokens=token_map[st.session_state.level])
        raw=r.choices[0].message.content; was_cut=r.choices[0].finish_reason=="length"
        final=smart_finish(raw, was_cut, st.session_state.level)
        b.empty(); st.markdown(f'<div class="chat-bubble ai-bubble">{final}</div>', unsafe_allow_html=True)
        current["messages"].append({"role":"assistant","content":final})
        st.rerun()
