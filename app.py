import streamlit as st, pickle, faiss, numpy as np, uuid, datetime, re, base64
from groq import Groq
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Cambridge AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at top, #1a2235 0%, #0e1117 70%); }
div[data-testid="stChatMessage"] { background: rgba(30,34,45,0.6)!important; backdrop-filter: blur(12px); border:1px solid rgba(255,255,255,0.08); border-radius:20px; }
div[data-testid="stChatInput"] { position: relative!important; }
div[data-testid="stChatInput"] > div { background: rgba(30,34,45,0.7)!important; border-radius:24px!important; border:1px solid rgba(255,255,255,0.1)!important; padding-right: 95px!important; }
div[data-testid="stChatInput"]:hover > div { border-color: rgba(77,166,255,0.5)!important; box-shadow: 0 0 18px rgba(77,166,255,0.35)!important; }
div[data-testid="stChatInput"]:focus-within > div { border:2px solid #4da6ff!important; box-shadow:none!important; }

/* 3 DOTS NEXT TO SEND ICON */
div[data-testid="stPopover"] {
    position: absolute!important;
    right: 50px!important; /* right next to send arrow */
    bottom: 6px!important;
    z-index: 9999!important;
}
div[data-testid="stPopover"] > button {
    background: rgba(255,255,255,0.08)!important;
    border:1px solid rgba(255,255,255,0.12)!important;
    border-radius:50%!important; width:32px!important; height:32px!important;
}
div[data-testid="stPopover"] > button:hover { background: rgba(77,166,255,0.25)!important; border-color:#4da6ff!important; }
</style>
""", unsafe_allow_html=True)

# PASTE HANDLER - Ctrl+V support
st.components.v1.html("""
<script>
document.addEventListener('paste', (e) => {
    const items = (e.clipboardData || window.clipboardData).items;
    for (let i=0; i<items.length; i++) {
        if (items[i].type.indexOf('image')!== -1) {
            const blob = items[i].getAsFile();
            // Show toast
            const div = document.createElement('div');
            div.innerText = '📷 Image pasted! Click Send to analyze';
            div.style.cssText='position:fixed;top:20px;left:50%;transform:translateX(-50%);background:#4da6ff;color:white;padding:10px 20px;border-radius:20px;z-index:99999;';
            document.body.appendChild(div);
            setTimeout(()=>div.remove(),2500);
        }
    }
});
</script>
""", height=0)

if "chats" not in st.session_state: st.session_state.chats={}
if "current_chat" not in st.session_state:
    nid=str(uuid.uuid4())[:8]
    st.session_state.current_chat=nid
    st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"Hey! Ask anything or 📷 paste picture with Ctrl+V"}]}
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

def fix(text):
    return text.replace(r'\[','$$').replace(r'\]','$$').replace('○','-')

with st.sidebar:
    st.markdown("## 🎓 Cambridge AI")
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        nid=str(uuid.uuid4())[:8]; st.session_state.current_chat=nid
        st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"Hey! Ask or paste image"}]}
        st.rerun()
    st.markdown("### 🕓 Old Chats")
    for cid,cdata in reversed(list(st.session_state.chats.items())):
        if st.button(f"{cdata['title'][:26]}", key=f"chat_{cid}", use_container_width=True):
            st.session_state.current_chat=cid; st.rerun()

current=st.session_state.chats[st.session_state.current_chat]
st.title("Cambridge AI")
st.caption(f"by Keane Moyo • {st.session_state.level} • Tip: Copy image then Ctrl+V in textbox")

for m in current["messages"]:
    with st.chat_message(m["role"]):
        if "image_bytes" in m: st.image(m["image_bytes"], width=350)
        st.markdown(fix(m["content"]))

# LEVEL
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

# --- 3 DOT MENU NEXT TO SEND ---
with st.popover("⋮", help="Upload picture"):
    st.markdown("**Add**")
    pic = st.file_uploader("📤 Upload picture", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
    cam = st.camera_input("📷 Camera")
    st.caption("Or just copy image & press Ctrl+V in textbox below")
    # Store in temp
    if pic: st.session_state["temp_pic"]=pic
    if cam: st.session_state["temp_pic"]=cam

# CHAT INPUT WITH FILE + PASTE SUPPORT
prompt_data = st.chat_input("ask or paste image (Ctrl+V)...", accept_file=True, file_type=["jpg","jpeg","png","webp"])

# Get image from 3 possible places: chat_input file, temp_pic from 3 dots, or paste
b64=None
image_bytes_for_display=None

if prompt_data:
    text = prompt_data.text if hasattr(prompt_data, 'text') else str(prompt_data)
    files = prompt_data.files if hasattr(prompt_data, 'files') else []

    # 1. File from chat_input (includes Ctrl+V paste!)
    if len(files)>0:
        image_bytes_for_display=files[0].getvalue()
        b64=to_b64(files[0])
        if not text: text="Explain / solve this picture in detail"

    # 2. File from 3-dot menu
    elif "temp_pic" in st.session_state and st.session_state["temp_pic"]:
        f=st.session_state["temp_pic"]
        image_bytes_for_display=f.getvalue() if hasattr(f,'getvalue') else f.read() if hasattr(f,'read') else None
        # rewind if file_uploader
        if hasattr(f,'seek'): f.seek(0)
        b64=to_b64(f)
        if not text: text="Explain this picture"

    if text:
        if current["title"]=="New Chat": current["title"]=text[:35]
        umsg={"role":"user","content":text}
        if image_bytes_for_display: umsg["image_bytes"]=image_bytes_for_display
        current["messages"].append(umsg)

        with st.chat_message("assistant"):
            ph=st.empty(); ph.markdown("👁️ Looking at picture...")
            token_map={"Simple":400,"Moderate":750,"Best":1000}
            q_emb=embed_model.encode([text]); D,I=index.search(np.array(q_emb).astype('float32'),3)
            context="\n\n".join([chunks[i] for i in I[0]])

            if b64:
                resp=client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {"role":"system","content": f"Cambridge AI tutor, level {st.session_state.level}, use **bold**, bullets, $$math$$. Context:{context}"},
                        {"role":"user","content":[
                            {"type":"text","text": text},
                            {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}}
                        ]}
                    ],
                    max_tokens=token_map[st.session_state.level]
                )
            else:
                resp=client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role":"system","content": f"Cambridge AI, level {st.session_state.level}. Context:{context}"},{"role":"user","content": text}],
                    max_tokens=token_map[st.session_state.level]
                )
            ans=fix(resp.choices[0].message.content)
            ph.empty(); st.markdown(ans)
            current["messages"].append({"role":"assistant","content":ans})

        if "temp_pic" in st.session_state: del st.session_state["temp_pic"]
        st.rerun()
