import streamlit as st, pickle, faiss, numpy as np, uuid, re, base64
from groq import Groq
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Cambridge AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at top, #1a2235 0%, #0e1117 70%); }
div[data-testid="stChatMessage"] { background: rgba(30,34,45,0.6)!important; backdrop-filter: blur(12px); border:1px solid rgba(255,255,255,0.08); border-radius:20px; }
div[data-testid="stChatInput"] > div { background: rgba(30,34,45,0.7)!important; border-radius:24px!important; border:1px solid rgba(255,255,255,0.1)!important; padding-left: 42px!important; }

/* ONE PLUS ONLY - INSIDE TEXTBOX */
div[data-testid="stPopover"] {
    position: fixed!important;
    bottom: 16px!important;
    left: 20px!important;
    z-index: 99999!important;
}
div[data-testid="stPopover"] > button {
    background: transparent!important; border: none!important;
    font-size: 22px!important; color: white!important;
    width: 30px!important; height: 30px!important;
}
</style>
""", unsafe_allow_html=True)

if "chats" not in st.session_state:
    st.session_state.chats={}
    nid=str(uuid.uuid4())[:8]
    st.session_state.current_chat=nid
    st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"Hey! Tap + for picture 📷"}]}
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

# Sidebar & chats
current=st.session_state.chats[st.session_state.current_chat]
st.title("Cambridge AI")
for m in current["messages"]:
    with st.chat_message(m["role"]):
        if "image_bytes" in m: st.image(m["image_bytes"], width=300)
        st.markdown(m["content"])

# Levels
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

# Preview when picture added (shows as uploaded in textbox)
if st.session_state.pending_img:
    st.image(st.session_state.pending_img, width=100)
    if st.button("❌ Remove picture"):
        st.session_state.pending_img=None; st.session_state.pending_b64=None; st.rerun()

# THE ONLY PLUS - this will be the + you circled
with st.popover("+"):
    st.markdown("**Add picture**")
    up = st.file_uploader("Upload from gallery", type=["jpg","jpeg","png","webp"])
    st.caption("JPG, PNG, WEBP")
    st.divider()
    st.write("📷 Shoot with camera")
    cam = st.camera_input("Take Photo")
    if up:
        st.session_state.pending_img=up
        st.session_state.pending_b64=to_b64(up)
        st.rerun()
    if cam:
        st.session_state.pending_img=cam
        st.session_state.pending_b64=to_b64(cam)
        st.rerun()

# Textbox at bottom - NO accept_file so only our + shows
prompt = st.chat_input("ask or paste image (Ctrl+V)...")

if prompt:
    text=prompt
    b64=st.session_state.pending_b64
    img_bytes=st.session_state.pending_img.getvalue() if st.session_state.pending_img and hasattr(st.session_state.pending_img,'getvalue') else None

    if text:
        if current["title"]=="New Chat": current["title"]=text[:35]
        umsg={"role":"user","content":text}
        if img_bytes: umsg["image_bytes"]=img_bytes
        current["messages"].append(umsg)

        with st.chat_message("assistant"):
            ph=st.empty(); ph.markdown("👁️ Analyzing..." if b64 else "💭...")
            q_emb=embed_model.encode([text]); D,I=index.search(np.array(q_emb).astype('float32'),3)
            context="\n\n".join([chunks[i] for i in I[0]])
            tok={"Simple":400,"Moderate":750,"Best":1000}
            if b64:
                resp=client.chat.completions.create(model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role":"system","content": f"Level {st.session_state.level}. Context:{context}"},
                              {"role":"user","content":[{"type":"text","text":text},{"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}}]}],
                    max_tokens=tok[st.session_state.level])
            else:
                resp=client.chat.completions.create(model="openai/gpt-oss-20b",
                    messages=[{"role":"system","content": f"Level {st.session_state.level}. Context:{context}"},{"role":"user","content":text}],
                    max_tokens=tok[st.session_state.level])
            ans=resp.choices[0].message.content
            ph.empty(); st.markdown(ans)
            current["messages"].append({"role":"assistant","content":ans})
        st.session_state.pending_img=None; st.session_state.pending_b64=None
        st.rerun()
