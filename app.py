import streamlit as st, pickle, faiss, numpy as np, uuid, re, base64
from groq import Groq
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Cambridge AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at top, #1a2235 0%, #0e1117 70%); }
div[data-testid="stChatMessage"] { background: rgba(30,34,45,0.6)!important; backdrop-filter: blur(12px); border:1px solid rgba(255,255,255,0.08); border-radius:20px; }
div[data-testid="stChatInput"] > div { background: rgba(30,34,45,0.7)!important; border-radius:24px!important; border:1px solid rgba(255,255,255,0.1)!important; padding-left: 45px!important; }

/* SINGLE PLUS INSIDE TEXTBOX - LEFT SIDE */
div[data-testid="stChatInput"] { position: relative!important; }
div[data-testid="stPopover"] {
    position: fixed!important;
    bottom: 18px!important;
    left: 22px!important;
    z-index: 99999!important;
}
div[data-testid="stPopover"] > button {
    background: transparent!important;
    border: none!important;
    font-size: 22px!important;
    font-weight: 300!important;
    color: white!important;
    width: 32px!important; height: 32px!important;
    box-shadow: none!important;
}
div[data-testid="stPopover"] > button:hover { background: rgba(255,255,255,0.1)!important; border-radius:50%!important; }
</style>
""", unsafe_allow_html=True)

# Ctrl+V paste support
st.components.v1.html("<script>document.addEventListener('paste',e=>{for(let i of e.clipboardData.items){if(i.type.includes('image')){let d=document.createElement('div');d.innerText='📷 Image pasted!';d.style.cssText='position:fixed;top:20px;left:50%;transform:translateX(-50%);background:#4da6ff;color:white;padding:8px 16px;border-radius:20px;z-index:99999;';document.body.appendChild(d);setTimeout(()=>d.remove(),2000);}}});</script>", height=0)

if "chats" not in st.session_state: st.session_state.chats={}
if "current_chat" not in st.session_state:
    nid=str(uuid.uuid4())[:8]
    st.session_state.current_chat=nid
    st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"Hey! Ask anything or tap + for picture 📷"}]}
if "level" not in st.session_state: st.session_state.level="Simple"
if "pending_img" not in st.session_state: st.session_state.pending_img=None
if "pending_b64" not in st.session_state: st.session_state.pending_b64=None

client=Groq(api_key=st.secrets["GROQ_API_KEY"])
@st.cache_resource
def load_brain():
    from sentence_transformers import SentenceTransformer
    chunks=pickle.load(open("chunks.pkl","rb"))
    index=faiss.read_index("model.faiss")
    model=SentenceTransformer('all-MiniLM-L6-v2')
    return chunks,index,model
chunks,index,embed_model=load_brain()

def to_b64(f):
    img=Image.open(f)
    if max(img.size)>1024: img.thumbnail((1024,1024))
    buf=BytesIO(); img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()
def fix(t): return t.replace(r'\[','$$').replace(r'\]','$$').replace('○','-')

with st.sidebar:
    st.markdown("## 🎓 Cambridge AI")
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        nid=str(uuid.uuid4())[:8]; st.session_state.current_chat=nid
        st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"Hey!"}]}
        st.session_state.pending_img=None; st.rerun()
    for cid,cdata in reversed(list(st.session_state.chats.items())):
        if st.button(f"{cdata['title'][:26]}", key=f"chat_{cid}", use_container_width=True):
            st.session_state.current_chat=cid; st.rerun()

current=st.session_state.chats[st.session_state.current_chat]
st.title("Cambridge AI")
st.caption(f"by Keane Moyo • {st.session_state.level}")

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

# Preview if image selected (shows above textbox like you wanted)
if st.session_state.pending_img:
    with st.container():
        st.image(st.session_state.pending_img, width=120)
        col1,col2=st.columns([1,5])
        with col1:
            if st.button("❌ Remove"):
                st.session_state.pending_img=None; st.session_state.pending_b64=None; st.rerun()
        with col2: st.caption("Image will be sent with your next message")

# THIS IS THE ONLY PLUS - INSIDE TEXTBOX AT BOTTOM
with st.popover("+"):
    st.markdown("**Add picture**")
    up = st.file_uploader("📤 Upload from gallery", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
    st.caption("200MB per file • JPG, PNG, WEBP")
    st.divider()
    st.markdown("📷 **Shoot with camera**")
    cam = st.camera_input("Take Photo", label_visibility="collapsed")
    st.caption("You can also Ctrl+V paste image")
    if up:
        st.session_state.pending_img=up
        st.session_state.pending_b64=to_b64(up)
        st.rerun()
    if cam:
        st.session_state.pending_img=cam
        st.session_state.pending_b64=to_b64(cam)
        st.rerun()

# TEXTBOX - ALWAYS AT BOTTOM
prompt = st.chat_input("ask or paste image (Ctrl+V)...", accept_file=True, file_type=["jpg","jpeg","png","webp"])

if prompt:
    text = prompt.text if hasattr(prompt, 'text') else str(prompt)
    files = prompt.files if hasattr(prompt, 'files') else []
    b64 = st.session_state.pending_b64
    img_bytes = None

    if len(files)>0:
        img_bytes=files[0].getvalue()
        b64=to_b64(files[0])
        if not text: text="Explain this picture"
    elif st.session_state.pending_img:
        if hasattr(st.session_state.pending_img,'getvalue'):
            img_bytes=st.session_state.pending_img.getvalue()
        if not text: text="Explain this picture"

    if text:
        if current["title"]=="New Chat": current["title"]=text[:35]
        umsg={"role":"user","content":text}
        if img_bytes: umsg["image_bytes"]=img_bytes
        current["messages"].append(umsg)

        with st.chat_message("assistant"):
            ph=st.empty(); ph.markdown("👁️ Analyzing..." if b64 else "💭 Thinking...")
            token_map={"Simple":400,"Moderate":750,"Best":1000}
            q_emb=embed_model.encode([text]); D,I=index.search(np.array(q_emb).astype('float32'),3)
            context="\n\n".join([chunks[i] for i in I[0]])
            if b64:
                resp=client.chat.completions.create(model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role":"system","content": f"Cambridge tutor level {st.session_state.level}. Context:{context}"},
                              {"role":"user","content":[{"type":"text","text": text},{"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}}]}],
                    max_tokens=token_map[st.session_state.level])
            else:
                resp=client.chat.completions.create(model="openai/gpt-oss-20b",
                    messages=[{"role":"system","content": f"Level {st.session_state.level}. Context:{context}"},{"role":"user","content": text}],
                    max_tokens=token_map[st.session_state.level])
            ans=fix(resp.choices[0].message.content)
            ph.empty(); st.markdown(ans)
            current["messages"].append({"role":"assistant","content":ans})

        st.session_state.pending_img=None; st.session_state.pending_b64=None
        st.rerun()
