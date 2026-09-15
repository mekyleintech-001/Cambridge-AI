import streamlit as st, pickle, faiss, numpy as np, uuid, datetime, re, base64
from groq import Groq
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Cambridge AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at top, #1a2235 0%, #0e1117 70%); }
div[data-testid="stChatMessage"] { background: rgba(30,34,45,0.6)!important; backdrop-filter: blur(12px); border:1px solid rgba(255,255,255,0.08); border-radius:20px; }

/* TEXTBOX */
div[data-testid="stChatInput"] { position: relative!important; }
div[data-testid="stChatInput"] > div { background: rgba(30,34,45,0.7)!important; border-radius:24px!important; border:1px solid rgba(255,255,255,0.1)!important; }

/* PLUS BUTTON BEFORE TEXTBOX - left side */
div[data-testid="stPopover"]:has(button:contains("+")) {
    position: absolute!important;
    left: 8px!important;
    bottom: 8px!important;
    z-index: 9999!important;
}
.plus-popover button {
    background: rgba(255,255,255,0.1)!important;
    border-radius:50%!important; width:32px!important; height:32px!important;
    font-size:20px!important; font-weight:bold!important;
}
/* Preview of image inside textbox area */
.img-preview {
    display:flex; gap:8px; align-items:center;
    background: rgba(30,34,45,0.8); border:1px solid rgba(255,255,255,0.1);
    border-radius:12px; padding:8px; margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

# Ctrl+V paste listener
st.components.v1.html("""
<script>
document.addEventListener('paste', (e) => {
    const items = e.clipboardData.items;
    for (let i=0;i<items.length;i++){
        if(items[i].type.includes('image')){
            const d=document.createElement('div');
            d.innerText='📷 Image pasted - will upload in textbox!';
            d.style.cssText='position:fixed;top:20px;left:50%;transform:translateX(-50%);background:#4da6ff;color:white;padding:10px 20px;border-radius:20px;z-index:99999;';
            document.body.appendChild(d); setTimeout(()=>d.remove(),2500);
        }
    }
});
</script>
""", height=0)

if "chats" not in st.session_state: st.session_state.chats={}
if "current_chat" not in st.session_state:
    nid=str(uuid.uuid4())[:8]
    st.session_state.current_chat=nid
    st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"Hey! Ask anything - or tap + to add picture 📷"}]}
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

def to_b64(file):
    img=Image.open(file)
    if max(img.size)>1024: img.thumbnail((1024,1024))
    buf=BytesIO(); img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def fix(t): return t.replace(r'\[','$$').replace(r'\]','$$').replace('○','-')

# SIDEBAR
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

# IMAGE PREVIEW INSIDE TEXTBOX AREA (like you asked)
if st.session_state.pending_img:
    st.markdown('<div class="img-preview">', unsafe_allow_html=True)
    col1,col2=st.columns([1,4])
    with col1: st.image(st.session_state.pending_img, width=80)
    with col2:
        st.caption("📷 Image ready - type your question and Send")
        if st.button("❌ Remove image"):
            st.session_state.pending_img=None; st.session_state.pending_b64=None; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- PLUS BUTTON BEFORE TEXTBOX (replaces the 3 dots) ---
# We place it just above chat_input so it looks like it's inside
left, right = st.columns([1,12])
with left:
    with st.popover("+", help="Add picture"):
        st.markdown("**Add picture**")
        upload = st.file_uploader("📤 Upload from gallery", type=["jpg","jpeg","png","webp"])
        cam = st.camera_input("📷 Shoot with camera")
        st.caption("Tip: You can also Ctrl+V paste image")
        if upload:
            st.session_state.pending_img=upload
            st.session_state.pending_b64=to_b64(upload)
            st.rerun()
        if cam:
            st.session_state.pending_img=cam
            st.session_state.pending_b64=to_b64(cam)
            st.rerun()

with right:
    prompt_data = st.chat_input("ask or paste image (Ctrl+V)...", accept_file=True, file_type=["jpg","jpeg","png","webp"])

# HANDLE SEND
if prompt_data:
    text = prompt_data.text if hasattr(prompt_data, 'text') else str(prompt_data)
    files = prompt_data.files if hasattr(prompt_data, 'files') else []

    b64 = st.session_state.pending_b64
    img_bytes = None

    # 1. File pasted / uploaded via + in chat_input itself (Ctrl+V)
    if len(files)>0:
        img_bytes=files[0].getvalue()
        b64=to_b64(files[0])
        if not text: text="Solve / explain this picture"

    # 2. File from our custom + popover (upload or camera)
    elif st.session_state.pending_img:
        f=st.session_state.pending_img
        if hasattr(f,'getvalue'): img_bytes=f.getvalue()
        # b64 already set
        if not text: text="Explain this picture taken by camera"

    if text:
        if current["title"]=="New Chat": current["title"]=text[:35]
        umsg={"role":"user","content":text}
        if img_bytes: umsg["image_bytes"]=img_bytes
        elif st.session_state.pending_img and hasattr(st.session_state.pending_img,'getvalue'):
            umsg["image_bytes"]=st.session_state.pending_img.getvalue()
        current["messages"].append(umsg)

        with st.chat_message("assistant"):
            ph=st.empty(); ph.markdown("👁️ Analyzing image..." if b64 else "💭 Thinking...")
            token_map={"Simple":400,"Moderate":750,"Best":1000}
            q_emb=embed_model.encode([text]); D,I=index.search(np.array(q_emb).astype('float32'),3)
            context="\n\n".join([chunks[i] for i in I[0]])

            if b64:
                resp=client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {"role":"system","content": f"Cambridge tutor, level {st.session_state.level}, bold, bullets, $$math$$. Context:{context}"},
                        {"role":"user","content":[{"type":"text","text": text},{"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}}]}
                    ],
                    max_tokens=token_map[st.session_state.level]
                )
            else:
                resp=client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role":"system","content": f"Level {st.session_state.level}. Context:{context}"},{"role":"user","content": text}],
                    max_tokens=token_map[st.session_state.level]
                )
            ans=fix(resp.choices[0].message.content)
            ph.empty(); st.markdown(ans)
            current["messages"].append({"role":"assistant","content":ans})

        st.session_state.pending_img=None; st.session_state.pending_b64=None
        st.rerun()
