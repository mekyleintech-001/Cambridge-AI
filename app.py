import streamlit as st, pickle, faiss, numpy as np, uuid, re, base64
from groq import Groq
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Cambridge AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at top, #1a2235 0%, #0e1117 70%); }
div[data-testid="stChatMessage"] { background: rgba(30,34,45,0.6)!important; backdrop-filter: blur(12px); border:1px solid rgba(255,255,255,0.08); border-radius:20px; }

/* ORIGINAL GLOWING TEXTBOX - RESTORED */
div[data-testid="stChatInput"] > div {
    background: rgba(30,34,45,0.7)!important;
    border-radius:24px!important;
    border:1px solid rgba(255,255,255,0.1)!important;
}
div[data-testid="stChatInput"]:hover > div {
    border-color: rgba(77,166,255,0.5)!important;
    box-shadow: 0 0 18px rgba(77,166,255,0.35)!important;
}
div[data-testid="stChatInput"]:focus-within > div {
    border:2px solid #4da6ff!important;
    box-shadow: none!important;
}
</style>
""", unsafe_allow_html=True)

if "chats" not in st.session_state:
    st.session_state.chats={}
    nid=str(uuid.uuid4())[:8]
    st.session_state.current_chat=nid
    st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"Hey! Ask or 📷 add picture with +"}]}
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

def to_b64(file):
    img=Image.open(file)
    if max(img.size)>1024: img.thumbnail((1024,1024))
    buf=BytesIO(); img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()
def fix(t): return t.replace(r'\[','$$').replace(r'\]','$$').replace('○','-')

with st.sidebar:
    st.markdown("## 🎓 Cambridge AI")
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        nid=str(uuid.uuid4())[:8]; st.session_state.current_chat=nid
        st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"Hey!"}]}
        st.rerun()

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

# ONLY ONE TEXTBOX - ORIGINAL GLOWING ONE - NO EXTRA BOX BELOW
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
        st.rerun()
