import streamlit as st
import uuid
import pickle
import faiss
import json
import base64
import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from groq import Groq
from io import BytesIO
from pathlib import Path

st.set_page_config(page_title="Kyle AI", page_icon="🎓", layout="wide")

# --- PERSISTENT HISTORY FOR LOGGED IN USERS ---
CHAT_FOLDER = Path("user_chats")
CHAT_FOLDER.mkdir(exist_ok=True)

def get_user_id():
    if st.user.is_logged_in:
        return st.user.email.replace('@','_at_').replace('.','_')
    return "guest"

def load_chats_for_user(uid):
    if uid == "guest":
        return None
    f = CHAT_FOLDER / f"{uid}.json"
    if f.exists():
        try:
            data = json.loads(f.read_text())
            # convert image_bytes from base64 if saved
            return data
        except:
            return None
    return None

def save_chats_for_user(uid, chats_dict):
    if uid == "guest":
        return
    f = CHAT_FOLDER / f"{uid}.json"
    # Don't save image_bytes (too big) - save only text
    clean_chats = {}
    for cid, chat in chats_dict.items():
        clean_msgs = []
        for m in chat["messages"]:
            clean_msgs.append({"role": m["role"], "content": m["content"]})
        clean_chats[cid] = {"title": chat["title"], "messages": clean_msgs}
    f.write_text(json.dumps(clean_chats))

# ----- GOOGLE LOGIN + GUEST -----
if "guest_mode" not in st.session_state:
    st.session_state.guest_mode = False

if not st.user.is_logged_in and not st.session_state.guest_mode:
    st.title("Welcome to Kyle AI 🤖")
    st.write("Please login to continue and save your chat history")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Continue with Google", use_container_width=True):
            st.login()
    with col2:
        if st.button("Continue as Guest", use_container_width=True):
            st.session_state.guest_mode = True
            st.rerun()
    st.stop()

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at top, #1a2235 0%, #0e1117 70%); }
div[data-testid="stChatMessage"] { background: rgba(30,34,45,0.6)!important; backdrop-filter: blur(12px); border:1px solid rgba(255,255,255,0.08); border-radius:20px; }
div[data-testid="stChatInput"] > div {
    background: rgba(30,34,45,0.7)!important; border-radius:24px!important; border:1px solid rgba(255,255,255,0.1)!important;
}
div[data-testid="stChatInput"]:hover > div { border-color: rgba(77,166,255,0.5)!important; box-shadow: 0 0 18px rgba(77,166,255,0.35)!important; }
div[data-testid="stChatInput"]:focus-within > div { border:2px solid #4da6ff!important; box-shadow: none!important; }
</style>
""", unsafe_allow_html=True)

uid = get_user_id()

# INIT CHATS
if "chats" not in st.session_state:
    loaded = load_chats_for_user(uid)
    if loaded:
        st.session_state.chats = loaded
        st.session_state.current_chat = list(loaded.keys())[0]
    else:
        nid=str(uuid.uuid4())[:8]
        st.session_state.current_chat=nid
        st.session_state.chats={nid:{"title":"New Chat","messages":[{"role":"assistant","content":"Hey! Ask or 📷 add picture with +"}]}}
    st.session_state.loaded_uid = uid

# If user switched from guest to logged in
if st.session_state.get("loaded_uid")!= uid:
    loaded = load_chats_for_user(uid)
    if loaded:
        st.session_state.chats = loaded
        st.session_state.current_chat = list(loaded.keys())[0]
    st.session_state.loaded_uid = uid

if "level" not in st.session_state: st.session_state.level="Simple"

client=Groq(api_key=st.secrets["GROQ_API_KEY"])
@st.cache_resource
def load_brain():
    chunks=pickle.load(open("chunks.pkl","rb"))
    index=faiss.read_index("model.faiss")
    model=SentenceTransformer('all-MiniLM-L6-v2')
    return chunks,index,model
chunks,index,embed_model=load_brain()

def to_b64(file):
    img=Image.open(file)
    if max(img.size)>1024: img.thumbnail((1024,1024))
    buf=BytesIO(); img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()
def fix(t): return t.replace(r'\[','$$').replace(r'\]','$$').replace('○','-')

with st.sidebar:
    st.markdown("## 🎓 Cambridge AI")
    if st.user.is_logged_in:
        st.write(f"Logged in as: {st.user.email}")
        if st.button("Logout", use_container_width=True):
            save_chats_for_user(uid, st.session_state.chats)
            st.logout()
    else:
        st.write("You are in Guest mode")
        st.caption("Login to save chat history")
        if st.button("Login with Google", use_container_width=True):
            st.login()
        if st.button("Exit Guest", use_container_width=True):
            st.session_state.guest_mode = False
            st.rerun()

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        nid=str(uuid.uuid4())[:8]; st.session_state.current_chat=nid
        st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"Hey!"}]}
        save_chats_for_user(uid, st.session_state.chats)
        st.rerun()
    st.divider()
    # List old chats
    for cid, chat in list(st.session_state.chats.items())[::-1][:10]:
        if st.button(chat["title"][:25], key=f"hist_{cid}", use_container_width=True):
            st.session_state.current_chat = cid
            st.rerun()

current=st.session_state.chats[st.session_state.current_chat]
st.title("Cambridge AI")
st.caption(f"by Keane Moyo • {st.session_state.level} • {uid}")

for m in current["messages"]:
    with st.chat_message(m["role"]):
        if "image_bytes" in m: st.image(m["image_bytes"], width=350)
        st.markdown(fix(m["content"]))

c1,c2,c3=st.columns(3)
with c1:
    if st.button("Simple 400", use_container_width=True, type="primary" if st.session_state.level=="Simple" else "secondary"):
        st.session_state.level="Simple"; st.rerun()
with c2:
    if st.button("Moderate 750", use_container_width=True, type="primary" if st.session_state.level=="Moderate" else "secondary"):
        st.session_state.level="Moderate"; st.rerun()
with c3:
    if st.button("Best 1000", use_container_width=True, type="primary" if st.session_state.level=="Best" else "secondary"):
        st.session_state.level="Best"; st.rerun()

prompt_data = st.chat_input("ask or paste image (Ctrl+V)...", accept_file=True, file_type=["jpg","jpeg","png","webp"])

if prompt_data:
    text = prompt_data.text if hasattr(prompt_data, 'text') else str(prompt_data)
    files = prompt_data.files if hasattr(prompt_data, 'files') else []
    b64=None
    img_bytes=None
    if len(files)>0:
        img_bytes=files[0].getvalue()
        b64=to_b64(files[0])
        if not text: text="Explain this picture in detail"
    if text:
        if current["title"]=="New Chat": current["title"]=text[:35]
        umsg={"role":"user","content":text}
        if img_bytes: umsg["image_bytes"]=img_bytes
        current["messages"].append(umsg)
        with st.chat_message("assistant"):
            ph=st.empty(); ph.markdown("👁️ Looking at picture..." if b64 else "💭 Thinking...")
            token_map={"Simple":400,"Moderate":750,"Best":1000}
            q_emb=embed_model.encode([text]); D,I=index.search(np.array(q_emb).astype('float32'),3)
            context="\n\n".join([chunks[i] for i in I[0]])
            if b64:
                resp=client.chat.completions.create(model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role":"system","content": f"Tutor {st.session_state.level}. Context:{context}"},
                              {"role":"user","content":[{"type":"text","text": text},{"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}}]}],
                    max_tokens=token_map[st.session_state.level])
            else:
                resp=client.chat.completions.create(model="openai/gpt-oss-20b",
                    messages=[{"role":"system","content": f"Level {st.session_state.level}. Context:{context}"},{"role":"user","content": text}],
                    max_tokens=token_map[st.session_state.level])
            ans=fix(resp.choices[0].message.content)
            ph.empty(); st.markdown(ans)
            current["messages"].append({"role":"assistant","content":ans})
        save_chats_for_user(uid, st.session_state.chats)
        st.rerun()
