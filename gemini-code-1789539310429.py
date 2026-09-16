import streamlit as st
import uuid, pickle, faiss, os, glob, re
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from supabase import create_client

def rebuild_file(base_name):
    if os.path.exists(base_name) and os.path.getsize(base_name) > 1000:
        try:
            if base_name.endswith(".pkl"):
                with open(base_name,'rb') as f: pickle.load(f)
            return True
        except:
            try: os.remove(base_name)
            except: pass
    parts = sorted(glob.glob(f"{base_name}.part*"))
    if not parts: return False
    try:
        with open(base_name,'wb') as out:
            for p in parts: out.write(open(p,'rb').read())
        return os.path.getsize(base_name) > 1000
    except: return False

rebuild_file("chunks.pkl")
rebuild_file("model.faiss")

def get_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
    except:
        url = "https://yehlwrdkccmbghzngwxg.supabase.co"
        key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InllaGx3cmRrY2NtYmdoem5nd3hnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk0NzIxNDYsImV4cCI6MjEwNTA0ODE0Nn0.Hrxj0yoa7wGbBP9QUAIC4fwIBQndVTg7WjpSi97uJ7E"
    return create_client(url, key)

supabase = get_supabase()

def get_user_id():
    if st.user.is_logged_in: return st.user.email.replace('@','_at_').replace('.','_')
    return "guest"

def load_chats_for_user(uid):
    if uid == "guest": return None
    try:
        res = supabase.table("chats").select("chats_json").eq("user_id", uid).execute()
        if res.data: return res.data[0]["chats_json"]
    except: pass
    return None

def save_chats_for_user(uid, chats_dict):
    if uid == "guest": return
    try:
        clean = {cid: {"title": chat["title"], "messages": [{"role": m["role"], "content": m["content"]} for m in chat["messages"]]} for cid, chat in chats_dict.items()}
        supabase.table("chats").upsert({"user_id": uid, "chats_json": clean}).execute()
    except: pass

st.set_page_config(page_title="Kyle AI", page_icon="🎓", layout="wide")

if "guest_mode" not in st.session_state: st.session_state.guest_mode = False
if not st.user.is_logged_in and not st.session_state.guest_mode:
    st.title("Welcome to Kyle AI 🤖")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Continue with Google", use_container_width=True): st.login()
    with c2:
        if st.button("Continue as Guest", use_container_width=True):
            st.session_state.guest_mode = True; st.rerun()
    st.stop()

st.markdown("""<style>.stApp{background:radial-gradient(ellipse at top,#1a2235 0%,#0e1117 70%);}div[data-testid="stChatMessage"]{background:rgba(30,34,45,0.6)!important;backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.08);border-radius:20px;}div[data-testid="stChatInput"]>div{background:rgba(30,34,45,0.7)!important;border-radius:24px!important;}</style>""", unsafe_allow_html=True)

uid = get_user_id()
if "chats" not in st.session_state:
    loaded = load_chats_for_user(uid)
    if loaded:
        st.session_state.chats = loaded
        st.session_state.current_chat = list(loaded.keys())[0]
    else:
        nid = str(uuid.uuid4())[:8]
        st.session_state.current_chat = nid
        st.session_state.chats = {nid: {"title": "New Chat", "messages": [{"role": "assistant", "content": "Hey! I'm Kyle AI - AS Level 9618/9709/9702/9231 expert. Ask me anything!"}]}}
    st.session_state.loaded_uid = uid

if st.session_state.get("loaded_uid")!= uid:
    loaded = load_chats_for_user(uid)
    if loaded:
        st.session_state.chats = loaded
        st.session_state.current_chat = list(loaded.keys())[0]
    st.session_state.loaded_uid = uid

if "level" not in st.session_state: st.session_state.level = "Best"

TEXT_MODELS = ["openai/gpt-oss-20b","openai/gpt-oss-120b","llama-3.3-70b-versatile","llama-3.1-8b-instant"]
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def call_groq_auto(messages, max_tokens):
    for mid in TEXT_MODELS:
        try: return client.chat.completions.create(model=mid, messages=messages, max_tokens=max_tokens), mid
        except: continue
    raise Exception("All models failed")

@st.cache_resource
def load_brain(_hash):
    chunks = pickle.load(open("chunks.pkl","rb"))
    index = faiss.read_index("model.faiss")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return chunks, index, model

try:
    chunks, index, embed_model = load_brain(os.path.getmtime("chunks.pkl") if os.path.exists("chunks.pkl") else 0)
except Exception as e:
    st.error(f"Brain files missing: {e}")
    st.stop()

def fix(t):
    def clean_sql(m):
        inner = m.group(1)
        if "\\text{" in inner and any(k in inner.upper() for k in ["FROM","JOIN","SELECT","WHERE","CUSTOMER","BORROW","ON"]):
            cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", inner)
            cleaned = cleaned.replace("\\"," ").replace("{","").replace("}","")
            cleaned = re.sub(r"\s+"," ", cleaned).strip()
            return f"`{cleaned}`"
        return f"$${inner}$$"     t = re.sub(r"\$\$(.*?)\$\$", clean_sql, t, flags=re.DOTALL)     t = t.replace("$$6pt","").replace("$$8pt","").replace("$$12pt","")
    t = t.replace(r"\[","$$").replace(r"\]","$$").replace(r"\(","$").replace(r"\)","$")     t = t.replace("$$$$","$$").replace("○","- ").replace("•","- ")     return t  def is_greeting(t): return len(t.strip()) < 25 and any(g in t.lower() for g in ["hi","hie","hello","hey","thanks","yo","morning","ok"])  def get_system_prompt(level, context):     inst = {         "Simple":"SHORT 3000 tokens. Direct M1 A1 only.",         "Moderate":"DETAILED 5000 tokens. Explain steps.",         "Best":"BEST MAX 8192 tokens. Full mark scheme: start **Subject detected: XXXX - Name**, paper/year, M1 A1 B1 FT bold, full explanation, LaTeX $$...$$ MATH ONLY, common mistakes, boxed final."     }[level]     return f"""You are Kyle AI - Cambridge AS Level ONLY (9618/9709/9702/9231).  STRICT RULES: 1. AS SYLLABUS ONLY. If question is NOT Cambridge AS, reply: "I only answer Cambridge AS Level 9618 / 9709 / 9702 / 9231. Please ask a Cambridge AS question." 2. ALWAYS remember previous messages in this chat. If user says "explain that again", "part b", "follow up", relate to last question. 3. LATEX: MATH ONLY $$x^2$$. NEVER put SQL/code in $$ or \\text{{}}.
4. SQL: ALWAYS use ```sql block, never $$.
5. {inst}

CONTEXT (AS past papers):
{context}
"""

with st.sidebar:
    st.markdown("## 🎓 Kyle AI")
    st.caption(f"📚 {len(chunks)} chunks • AS Only • With Memory")
    if st.user.is_logged_in:
        st.caption(f"👤 {st.user.email}")
        if st.button("Logout", use_container_width=True):
            save_chats_for_user(uid, st.session_state.chats); st.logout()
    else:
        if st.button("Login with Google", use_container_width=True): st.login()
        if st.button("Exit Guest", use_container_width=True):
            st.session_state.guest_mode=False; st.rerun()
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        nid = str(uuid.uuid4())[:8]
        st.session_state.chats[nid] = {"title":"New Chat","messages":[{"role":"assistant","content":"New chat started! I remember our chat now."}]}
        st.session_state.current_chat = nid
        save_chats_for_user(uid, st.session_state.chats); st.rerun()
    if st.button("🗑️ Delete Current Chat", use_container_width=True):
        cid = st.session_state.current_chat
        if len(st.session_state.chats) > 1:
            del st.session_state.chats[cid]
            st.session_state.current_chat = list(st.session_state.chats.keys())[0]
        else:
            nid = str(uuid.uuid4())[:8]
            st.session_state.chats = {nid: {"title":"New Chat","messages":[{"role":"assistant","content":"New chat!"}]}}
            st.session_state.current_chat = nid
        save_chats_for_user(uid, st.session_state.chats); st.rerun()
    if st.button("🗑️ Delete All Chats", use_container_width=True):
        nid = str(uuid.uuid4())[:8]
        st.session_state.chats = {nid: {"title":"New Chat","messages":[{"role":"assistant","content":"All chats deleted. New chat!"}]}}
        st.session_state.current_chat = nid
        save_chats_for_user(uid, st.session_state.chats); st.rerun()
    st.divider()
    st.caption("Recent Chats")
    for cid in list(st.session_state.chats.keys())[::-1][:30]:
        chat = st.session_state.chats[cid]
        col1, col2 = st.columns([0.82,0.18])
        with col1:
            if st.button(chat["title"][:28], key=f"open_{cid}", use_container_width=True):
                st.session_state.current_chat = cid; st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{cid}", use_container_width=True):
                if len(st.session_state.chats) > 1:
                    del st.session_state.chats[cid]
                    if st.session_state.current_chat == cid:
                        st.session_state.current_chat = list(st.session_state.chats.keys())[0]
                else:
                    nid = str(uuid.uuid4())[:8]
                    st.session_state.chats = {nid: {"title":"New Chat","messages":[{"role":"assistant","content":"New chat!"}]}}
                    st.session_state.current_chat = nid
                save_chats_for_user(uid, st.session_state.chats); st.rerun()

current = st.session_state.chats[st.session_state.current_chat]
st.title("Cambridge AI - AS Level")
st.caption(f"Level: {st.session_state.level} • {len(chunks)} chunks • Memory ON")

for m in current["messages"]:
    with st.chat_message(m["role"]): st.markdown(fix(m["content"]))

c1,c2,c3 = st.columns(3)
with c1:
    if st.button("Simple 3000", use_container_width=True, type="primary" if st.session_state.level=="Simple" else "secondary"):
        st.session_state.level="Simple"; st.rerun()
with c2:
    if st.button("Moderate 5000", use_container_width=True, type="primary" if st.session_state.level=="Moderate" else "secondary"):
        st.session_state.level="Moderate"; st.rerun()
with c3:
    if st.button("Best MAX", use_container_width=True, type="primary" if st.session_state.level=="Best" else "secondary"):
        st.session_state.level="Best"; st.rerun()

prompt = st.chat_input("Ask any Cambridge AS question...")
if prompt:
    if current["title"] == "New Chat": current["title"] = prompt[:35]
    current["messages"].append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        ph = st.empty()
        ph.markdown("🔍 Searching AS past papers...")

        # --- MEMORY: last 8 messages ---
        history = current["messages"][-10:-1] # last 8 before current prompt
        history_text = "\n".join([f"{m['role']}: {m['content'][:500]}" for m in history])

        # --- RAG SEARCH ---
        q_emb = embed_model.encode([prompt])
        D,I = index.search(np.array(q_emb).astype('float32'), 25)
        raw_all = [chunks[i] for i in I[0]]

        def get_year(t):
            yrs = re.findall(r"(20[1-2][0-9])", t)
            return max([int(y) for y in yrs]) if yrs else 0

        texts = [(c["text"] if isinstance(c, dict) else c) for c in raw_all]
        qp=[]; ms=[]
        for txt in texts:
            if any(a2 in txt.lower() for a2 in ["p3","paper 3","m2","s2","a2 level"]): continue
            if any(k in txt for k in ["M1","A1","B1","M0","FT"]): ms.append(txt)
            else: qp.append(txt)

        qp_s = sorted(qp, key=get_year, reverse=True)
        ms_s = sorted(ms, key=get_year, reverse=True)
        if ms_s:
            context = f"AS QP:\n{qp_s[:3]}\n\nAS MS:\n" + "\n---\n".join(ms_s[:6]) + f"\n\nCHAT HISTORY:\n{history_text}"
        else:
            context = "\n---\n".join(sorted(texts, key=get_year, reverse=True)[:6]) + f"\n\nCHAT HISTORY:\n{history_text}"

        # --- GREETING CHECK ---
        if is_greeting(prompt):
            msgs = [{"role":"system","content":"You are Kyle AI. Greeting only, friendly short. AS Level only."}, {"role":"user","content":prompt}]
            resp,_ = call_groq_auto(msgs, 300)
        else:
            sys_prompt = get_system_prompt(st.session_state.level, context)
            msgs = [{"role":"system","content":sys_prompt}]
            # Add memory messages
            for m in history[-6:]:
                msgs.append({"role": m["role"], "content": m["content"]})
            msgs.append({"role":"user","content":prompt})
            token_map = {"Simple":3000,"Moderate":5000,"Best":8192}
            resp,_ = call_groq_auto(msgs, token_map[st.session_state.level])

        ans = fix(resp.choices[0].message.content)
        ph.empty(); st.markdown(ans)
        current["messages"].append({"role":"assistant","content":ans})

    save_chats_for_user(uid, st.session_state.chats)
    st.rerun()