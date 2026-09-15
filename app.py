import streamlit as st
import uuid, pickle, faiss, os, glob, re
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from supabase import create_client

# --- Load FAISS ---
if not os.path.exists("model.faiss"):
    part_files = sorted(glob.glob("model.faiss.part*"))
    if part_files:
        with open("model.faiss", 'wb') as out:
            for p in part_files:
                out.write(open(p,'rb').read())

def get_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
    except:
        url = os.getenv("SUPABASE_URL", "https://yehlwrdkccmbghzngwxg.supabase.co")
        key = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InllaGx3cmRrY2NtYmdoem5nd3hnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk0NzIxNDYsImV4cCI6MjEwNTA0ODE0Nn0.Hrxj0yoa7wGbBP9QUAIC4fwIBQndVTg7WjpSi97uJ7E")
    return create_client(url, key)

supabase = get_supabase()

def get_user_id():
    if st.user.is_logged_in:
        return st.user.email.replace('@','_at_').replace('.','_')
    return "guest"

def load_chats_for_user(uid):
    if uid == "guest": return None
    try:
        res = supabase.table("chats").select("chats_json").eq("user_id", uid).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]["chats_json"]
    except: pass
    return None

def save_chats_for_user(uid, chats_dict):
    if uid == "guest": return
    try:
        clean_chats = {}
        for cid, chat in chats_dict.items():
            clean_msgs = [{"role": m["role"], "content": m["content"]} for m in chat["messages"]]
            clean_chats[cid] = {"title": chat["title"], "messages": clean_msgs}
        supabase.table("chats").upsert({"user_id": uid, "chats_json": clean_chats}).execute()
    except: pass

st.set_page_config(page_title="Kyle AI", page_icon="🎓", layout="wide")

if "guest_mode" not in st.session_state: st.session_state.guest_mode = False
if not st.user.is_logged_in and not st.session_state.guest_mode:
    st.title("Welcome to Kyle AI 🤖")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Continue with Google", use_container_width=True): st.login()
    with col2:
        if st.button("Continue as Guest", use_container_width=True): st.session_state.guest_mode=True; st.rerun()
    st.stop()

st.markdown("""<style>
.stApp { background: radial-gradient(ellipse at top, #1a2235 0%, #0e1117 70%); }
div[data-testid="stChatMessage"] { background: rgba(30,34,45,0.6)!important; backdrop-filter: blur(12px); border:1px solid rgba(255,255,255,0.08); border-radius:20px; }
div[data-testid="stChatInput"] > div { background: rgba(30,34,45,0.7)!important; border-radius:24px!important; border:1px solid rgba(255,255,255,0.1)!important; }
</style>""", unsafe_allow_html=True)

uid = get_user_id()
if "chats" not in st.session_state:
    loaded = load_chats_for_user(uid)
    if loaded:
        st.session_state.chats=loaded; st.session_state.current_chat=list(loaded.keys())[0]
    else:
        nid=str(uuid.uuid4())[:8]; st.session_state.current_chat=nid
        st.session_state.chats={nid:{"title":"New Chat","messages":[{"role":"assistant","content":"Hey! I'm Kyle AI - your Cambridge 9618/9709/9702/9231 expert. Ask me anything!"}]}}
    st.session_state.loaded_uid=uid

if st.session_state.get("loaded_uid")!=uid:
    loaded=load_chats_for_user(uid)
    if loaded: st.session_state.chats=loaded; st.session_state.current_chat=list(loaded.keys())[0]
    st.session_state.loaded_uid=uid

if "level" not in st.session_state: st.session_state.level="Best"

# === TEXT MODELS AUTO-TRY ===
TEXT_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

client=Groq(api_key=st.secrets["GROQ_API_KEY"])

def call_groq_auto(messages, max_tokens):
    for model_id in TEXT_MODELS:
        try:
            resp = client.chat.completions.create(model=model_id, messages=messages, max_tokens=max_tokens)
            return resp, model_id
        except Exception as e:
            print(f"❌ {model_id} failed: {e}")
            continue
    raise Exception("All models failed")

def get_chunks_hash():
    try: return os.path.getmtime("chunks.pkl")
    except: return 0

@st.cache_resource
def load_brain(file_hash):
    chunks=pickle.load(open("chunks.pkl","rb"))
    index=faiss.read_index("model.faiss")
    model=SentenceTransformer('all-MiniLM-L6-v2')
    return chunks,index,model

chunks,index,embed_model=load_brain(get_chunks_hash())

def fix(t):
    t = t.replace(r'\[', '$$').replace(r'\]', '$$')
    t = t.replace(r'\(', '$').replace(r'\)', '$')
    return t

def is_greeting(t):
    t=t.lower().strip()
    if len(t)>25: return False
    return any(g in t for g in ["hi","hie","hello","hey","thanks","yo","morning","ok"])

def get_system_prompt(level, context, greeting=False):
    if greeting:
        return "You are Kyle AI. User just greeted. Reply friendly, short, ask how you can help with Cambridge CS/Maths/Physics. Do NOT say Subject detected."

    token_instruction = {
        "Simple": "Give SHORT 3000 token answer. Concise, direct, M1 A1 points only.",
        "Moderate": "Give DETAILED 5000 token answer. Explain steps, show working.",
        "Best": "Give BEST MAX 8192 token answer. Full Cambridge marking scheme style: state paper, use M1 A1 B1 bold keywords, full explanation, common mistakes, LaTeX, final answer boxed."
    }[level]

    return f"""You are Kyle AI - Cambridge expert for 9618 CS, 9709 Maths, 9702 Physics, 9231 Further Maths.

STYLE: {token_instruction}

RULES:
- If greeting: NEVER say Subject detected.
- If academic: ALWAYS start with **Subject detected: [CODE] [Name]**
- SUBJECT DETECTION (priority):
    1. polar, r=f(theta), de Moivre, complex number, matrix, eigenvalue, hyperbolic, Maclaurin -> 9231 Further Maths (polar ALWAYS 9231, NEVER 9709)
    2. force, velocity, electric field, magnetic, quantum, wave -> 9702 Physics
    3. algorithm, database, SQL, network, binary, stack, OOP, logic gates -> 9618 CS
    4. else -> 9709 Maths

- RETRIEVAL: You are given SIMILAR QP + MARK SCHEME. Mention year like "From 2024 mark scheme".
- Use M1 A1 B1 FT bold.
- LaTeX: $$A = \\frac{{1}}{{2}} \\int_{{\\alpha}}^{{\\beta}} [f(\\theta)]^2 d\\theta$$ never write (theta)
- Be accurate to mark scheme.

CONTEXT (QP + MS LINKED, latest year first):
{context}
"""

with st.sidebar:
    st.markdown("## 🎓 Kyle AI")
    st.caption(f"📚 {len(chunks)} chunks loaded")
    if st.user.is_logged_in:
        st.write(f"👤 {st.user.email}")
        if st.button("Logout", use_container_width=True): save_chats_for_user(uid, st.session_state.chats); st.logout()
    else:
        if st.button("Login with Google", use_container_width=True): st.login()
        if st.button("Exit Guest", use_container_width=True): st.session_state.guest_mode=False; st.rerun()
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        nid=str(uuid.uuid4())[:8]; st.session_state.current_chat=nid
        st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"New chat started!"}]}
        save_chats_for_user(uid, st.session_state.chats); st.rerun()
    st.divider()
    st.caption("Chats")
    for cid, chat in list(st.session_state.chats.items())[::-1][:20]:
        if st.button(chat["title"][:28], key=f"hist_{cid}", use_container_width=True):
            st.session_state.current_chat=cid; st.rerun()

current=st.session_state.chats[st.session_state.current_chat]
st.title("Cambridge AI")
st.caption(f"Level: {st.session_state.level} • {len(chunks)} past paper chunks • Auto-try {len(TEXT_MODELS)} models")

for m in current["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(fix(m["content"]))

# Level buttons
c1,c2,c3=st.columns(3)
with c1:
    if st.button("Simple 3000", use_container_width=True, type="primary" if st.session_state.level=="Simple" else "secondary"):
        st.session_state.level="Simple"; st.rerun()
with c2:
    if st.button("Moderate 5000", use_container_width=True, type="primary" if st.session_state.level=="Moderate" else "secondary"):
        st.session_state.level="Moderate"; st.rerun()
with c3:
    if st.button("Best MAX 8192", use_container_width=True, type="primary" if st.session_state.level=="Best" else "secondary"):
        st.session_state.level="Best"; st.rerun()

prompt = st.chat_input("Ask any Cambridge question...")
if prompt:
    if current["title"]=="New Chat": current["title"]=prompt[:35]
    current["messages"].append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        ph=st.empty()
        token_map={"Simple":3000,"Moderate":5000,"Best":8192}

        if is_greeting(prompt):
            sys_prompt = get_system_prompt(st.session_state.level, "", greeting=True)
            msgs=[{"role":"system","content":sys_prompt},{"role":"user","content":prompt}]
            resp, used = call_groq_auto(msgs, 600)
            ans=fix(resp.choices[0].message.content)
            ph.empty(); st.markdown(ans)
            current["messages"].append({"role":"assistant","content":ans})
        else:
            ph.markdown("🔍 Searching past papers...")
            q_emb=embed_model.encode([prompt])
            D,I=index.search(np.array(q_emb).astype('float32'), 25)
            raw_all=[chunks[i] for i in I[0]]

            def get_y(t):
                yrs=re.findall(r'(20[1-2][0-9])', t)
                return max([int(y) for y in yrs]) if yrs else 0

            raw_str_list=[(c["text"] if isinstance(c, dict) else c) for c in raw_all]
            qp_like=[]; ms_like=[]
            for txt in raw_str_list:
                if any(k in txt for k in ["M1","A1","B1","M0","FT","mark scheme","MARK SCHEME"]):
                    ms_like.append(txt)
                else:
                    qp_like.append(txt)

            qp_sorted=sorted(qp_like, key=get_y, reverse=True)
            ms_sorted=sorted(ms_like, key=get_y, reverse=True)
            best_qp=qp_sorted[0] if qp_sorted else raw_str_list[0]

            ph.markdown("📄 Found QP + MS, answering...")

            if ms_sorted:
                context=f"SIMILAR QUESTION PAPER FOUND (most recent):\n{best_qp}\n\n---\n\nEQUIVALENT MARK SCHEMES (latest first):\n" + "\n\n---\n\n".join(ms_sorted[:7])
            else:
                context="\n\n---\n\n".join(sorted(raw_str_list, key=get_y, reverse=True)[:8])

            sys_prompt=get_system_prompt(st.session_state.level, context, greeting=False)
            msgs=[{"role":"system","content":sys_prompt},{"role":"user","content":prompt}]

            resp, used = call_groq_auto(msgs, token_map[st.session_state.level])
            ans=fix(resp.choices[0].message.content)
            ph.empty(); st.markdown(ans)
            current["messages"].append({"role":"assistant","content":ans})

    save_chats_for_user(uid, st.session_state.chats)
    st.rerun()
