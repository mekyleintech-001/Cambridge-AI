import streamlit as st, pickle, faiss, numpy as np, uuid, datetime, re
from groq import Groq

st.set_page_config(page_title="Cambridge AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at top, #1a2235 0%, #0e1117 70%); }
div[data-testid="stChatMessage"] {
    background: rgba(30, 34, 45, 0.6)!important;
    backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08);
    border-radius:20px; margin:8px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
div[data-testid="stChatInput"] > div {
    background: rgba(30,34,45,0.7)!important; border-radius:24px!important;
    border:1px solid rgba(255,255,255,0.1)!important;
}
div[data-testid="stChatInput"]:hover > div { border-color: rgba(77,166,255,0.5)!important; box-shadow: 0 0 18px rgba(77,166,255,0.35)!important; }
div[data-testid="stChatInput"]:focus-within > div { border:2px solid #4da6ff!important; box-shadow:none!important; }
.typing { display:flex; gap:6px; padding:12px; }
.dot { width:9px; height:9px; background:#FFD700; border-radius:50%; animation: bounce 1.4s infinite; }
@keyframes bounce { 0%,80%,100%{transform:scale(0.7);opacity:0.5} 40%{transform:scale(1.2);opacity:1} }
</style>
""", unsafe_allow_html=True)

if "chats" not in st.session_state: st.session_state.chats={}
if "current_chat" not in st.session_state:
    nid=str(uuid.uuid4())[:8]
    st.session_state.current_chat=nid
    st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"Hey! Ask me anything 📚"}]}
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

def fix_formatting(text):
    # 1. Convert \[ \] to $$ $$ and \( \) to $ $
    text = text.replace(r'\[', '$$').replace(r'\]', '$$')
    text = text.replace(r'\(', '$').replace(r'\)', '$')
    # Fix double escaping from model
    text = re.sub(r'\\\[', '$$', text)
    text = re.sub(r'\\\]', '$$', text)
    text = re.sub(r'\\\( ', '$', text)
    text = re.sub(r'\\\)', '$', text)

    # 2. Fix \text{NAND} -> \text{NAND} is ok but ensure math mode works, keep it
    # 3. Remove stray circle bullets ○ and fix nested lists
    text = text.replace('○', '-')
    text = text.replace('●', '-')

    # 4. Fix ** inside code that breaks
    # 5. Clean hanging examples
    lines=text.split('\n')
    junk=[r'^\s*Examples?\s*:?\s*$', r'^\s*\d+[\.\)]\s*$', r'^\s*[-•*]\s*$']
    while lines:
        last=lines[-1].strip()
        if any(re.match(p,last,re.IGNORECASE) for p in junk) and len(last)<25:
            lines.pop()
        else: break
    text='\n'.join(lines)

    # 6. Ensure display math is $$ on separate lines for Streamlit
    text = re.sub(r'\$\$\s*', '\n\n$$\n', text)
    text = re.sub(r'\s*\$\$', '\n$$\n\n', text)
    # Clean double $$$$
    text = text.replace('$$\n\n$$\n\n$$\n', '$$\n').replace('$$\n\n$$', '$$')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def smart_finish(text, was_cut, level):
    text=fix_formatting(text)
    if not was_cut: return text
    last_good=max(text.rfind('. '), text.rfind('.\n'), text.rfind('$$\n'))
    if last_good>len(text)*0.5 and last_good!=-1:
        if text[last_good]=='.': text=text[:last_good+1]
        else: text=text[:last_good+1]
    if level=="Simple": text+="\n\n> 💡 Core covered. Switch to **Moderate** or **Best** for more."
    return text.strip()

with st.sidebar:
    st.markdown("## 🎓 Cambridge AI")
    st.caption("by Keane Moyo")
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        nid=str(uuid.uuid4())[:8]; st.session_state.current_chat=nid
        st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"Hey! Ask me anything 📚"}]}
        st.rerun()
    st.markdown("### 🕓 Old Chats")
    for cid,cdata in reversed(list(st.session_state.chats.items())):
        if st.button(f"{'🟡 ' if cid==st.session_state.current_chat else '💬 '}{cdata['title'][:26]}", key=f"chat_{cid}", use_container_width=True):
            st.session_state.current_chat=cid; st.rerun()

current=st.session_state.chats[st.session_state.current_chat]
st.title("Cambridge AI")
st.caption(f"by Keane Moyo • Level: {st.session_state.level}")

for m in current["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(fix_formatting(m["content"]))

st.markdown("---")
c1,c2,c3,c4 = st.columns([1,1,1,2])
with c1:
    if st.button("Simple 400", use_container_width=True, type="primary" if st.session_state.level=="Simple" else "secondary"):
        st.session_state.level="Simple"; st.rerun()
with c2:
    if st.button("Moderate 750", use_container_width=True, type="primary" if st.session_state.level=="Moderate" else "secondary"):
        st.session_state.level="Moderate"; st.rerun()
with c3:
    if st.button("Best 1000", use_container_width=True, type="primary" if st.session_state.level=="Best" else "secondary"):
        st.session_state.level="Best"; st.rerun()
with c4:
    st.caption(f"**{st.session_state.level}**")

query = st.chat_input("ask about anything...")
if query:
    if current["title"]=="New Chat": current["title"]=query[:35]
    current["messages"].append({"role":"user","content":query})
    token_map={"Simple":400,"Moderate":750,"Best":1000}
    prompts={
        "Simple":"You are Cambridge AI. Use clear markdown. **Bold** key terms. Use bullet lists with '- '. For math, use $ inline like $NAND(A,B) = \\neg(A \\land B)$ and $$ on own lines for big formulas like $$File\\ size = \\frac{width \\times height \\times bpp}{8}$$. Do NOT use \\[ \\] or \\( \\). Keep complete within 350 tokens. Do not start sections you can't finish.",
        "Moderate":"You are Cambridge AI. Detailed answer with headings ##, bold terms, bullet lists '- ', and proper math: inline $...$ and display $$...$$ on own lines. Do NOT use \\[ \\]. Complete within 700 tokens.",
        "Best":"You are Cambridge AI. Full comprehensive answer. Use ## headings, **bold**, bullet lists '- ', and math with $$...$$ for display. Example: $$NAND(A,B) = \\neg(A \\land B)$$. Include tables if needed. Complete within 1000 tokens."
    }
    with st.chat_message("assistant"):
        ph=st.empty(); ph.markdown('<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>', unsafe_allow_html=True)
        q_emb=embed_model.encode([query]); D,I=index.search(np.array(q_emb).astype('float32'),3)
        context="\n\n".join([chunks[i] for i in I[0]])
        r=client.chat.completions.create(model="openai/gpt-oss-20b", messages=[{"role":"system","content":f"{prompts[st.session_state.level]} Context:{context}"},{"role":"user","content":query}], max_tokens=token_map[st.session_state.level])
        raw=r.choices[0].message.content; was_cut=r.choices[0].finish_reason=="length"
        final=smart_finish(raw, was_cut, st.session_state.level)
        ph.empty(); st.markdown(final)
        current["messages"].append({"role":"assistant","content":final})
        st.rerun()
