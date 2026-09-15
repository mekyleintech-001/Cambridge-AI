import streamlit as st, pickle, faiss, numpy as np, uuid, datetime, re, base64
from groq import Groq
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Cambridge AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at top, #1a2235 0%, #0e1117 70%); }
div[data-testid="stChatMessage"] { background: rgba(30,34,45,0.6)!important; backdrop-filter: blur(12px); border:1px solid rgba(255,255,255,0.08); border-radius:20px; }
div[data-testid="stChatInput"] > div { background: rgba(30,34,45,0.7)!important; border-radius:24px!important; }
</style>
""", unsafe_allow_html=True)

if "chats" not in st.session_state: st.session_state.chats={}
if "current_chat" not in st.session_state:
    nid=str(uuid.uuid4())[:8]
    st.session_state.current_chat=nid
    st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"Hey! Send text or 📷 picture of a question"}]}
if "level" not in st.session_state: st.session_state.level="Simple"
if "pending_image" not in st.session_state: st.session_state.pending_image=None

client=Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def load_brain():
    from sentence_transformers import SentenceTransformer
    chunks=pickle.load(open("chunks.pkl","rb"))
    index=faiss.read_index("model.faiss")
    model=SentenceTransformer('all-MiniLM-L6-v2')
    return chunks,index,model
chunks,index,embed_model=load_brain()

def fix_formatting(text):
    text=text.replace(r'\[','$$').replace(r'\]','$$')
    text=re.sub(r'\\\[','$$',text); re.sub(r'\\\]','$$',text)
    text=text.replace('○','-')
    return text.strip()

def image_to_base64(img_file):
    img=Image.open(img_file)
    if max(img.size)>1024: img.thumbnail((1024,1024))
    buf=BytesIO(); img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

# SIDEBAR WITH CAMERA
with st.sidebar:
    st.markdown("## 🎓 Cambridge AI")
    st.caption("by Keane Moyo")
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        nid=str(uuid.uuid4())[:8]; st.session_state.current_chat=nid
        st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"Hey! Send text or 📷 picture"}]}
        st.session_state.pending_image=None; st.rerun()

    st.markdown("### 📷 Add Picture")
    cam = st.camera_input("Take picture of question")
    up = st.file_uploader("Or upload", type=["jpg","jpeg","png","webp"])

    if cam: st.session_state.pending_image=cam
    if up: st.session_state.pending_image=up

    if st.session_state.pending_image:
        st.image(st.session_state.pending_image, caption="Will be sent with next message", use_container_width=True)
        if st.button("❌ Remove picture"):
            st.session_state.pending_image=None; st.rerun()

    st.markdown("### 🕓 Old Chats")
    for cid,cdata in reversed(list(st.session_state.chats.items())):
        if st.button(f"{'🟡 ' if cid==st.session_state.current_chat else '💬 '}{cdata['title'][:26]}", key=f"chat_{cid}", use_container_width=True):
            st.session_state.current_chat=cid; st.rerun()

current=st.session_state.chats[st.session_state.current_chat]
st.title("Cambridge AI")
st.caption(f"by Keane Moyo • Level: {st.session_state.level}")

# DISPLAY CHAT WITH IMAGES
for m in current["messages"]:
    with st.chat_message(m["role"]):
        if "image" in m: st.image(m["image"], width=350)
        st.markdown(fix_formatting(m["content"]))

# LEVEL BUTTONS - CLEARLY VISIBLE
st.markdown("---")
c1,c2,c3 = st.columns(3)
with c1:
    if st.button("Simple 400", use_container_width=True, type="primary" if st.session_state.level=="Simple" else "secondary"):
        st.session_state.level="Simple"; st.rerun()
with c2:
    if st.button("Moderate 750", use_container_width=True, type="primary" if st.session_state.level=="Moderate" else "secondary"):
        st.session_state.level="Moderate"; st.rerun()
with c3:
    if st.button("Best 1000", use_container_width=True, type="primary" if st.session_state.level=="Best" else "secondary"):
        st.session_state.level="Best"; st.rerun()

# INPUT
query = st.chat_input("ask about anything, or attach a picture above...")

if query or (st.session_state.pending_image and query is None):
    # Handle case where only image sent
    if not query: query = "Explain this picture / solve this question in detail"

    if current["title"]=="New Chat": current["title"]=query[:35]

    # Prepare user message with image
    user_msg={"role":"user","content":query}
    b64=None
    if st.session_state.pending_image:
        b64=image_to_base64(st.session_state.pending_image)
        user_msg["image"]=st.session_state.pending_image

    current["messages"].append(user_msg)

    with st.chat_message("user"):
        if st.session_state.pending_image: st.image(st.session_state.pending_image, width=350)
        st.markdown(query)

    with st.chat_message("assistant"):
        ph=st.empty(); ph.markdown("🔍 Analyzing...")
        try:
            q_emb=embed_model.encode([query]); D,I=index.search(np.array(q_emb).astype('float32'),3)
            context="\n\n".join([chunks[i] for i in I[0]])
            token_map={"Simple":400,"Moderate":750,"Best":1000}

            if b64:
                # VISION MODEL FOR PICTURES
                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct", # Groq vision model【3652013737970536682†L62-L65】
                    messages=[
                        {"role":"system","content": f"You are Cambridge AI tutor. Explain clearly with **bold**, bullets, and $$math$$. Level {st.session_state.level}. Context:{context}"},
                        {"role":"user","content": [
                            {"type":"text","text": query},
                            {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}}
                        ]}
                    ],
                    max_tokens=token_map[st.session_state.level]
                )
            else:
                # TEXT ONLY
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role":"system","content": f"You are Cambridge AI. Level {st.session_state.level}. Use **bold**, bullets, $$math$$. Context:{context}"},
                        {"role":"user","content": query}
                    ],
                    max_tokens=token_map[st.session_state.level]
                )

            ans=fix_formatting(response.choices[0].message.content)
            ph.empty(); st.markdown(ans)
            current["messages"].append({"role":"assistant","content":ans})

        except Exception as e:
            ph.empty(); st.error(f"Error: {e}")

    st.session_state.pending_image=None
    st.rerun()
