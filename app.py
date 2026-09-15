import streamlit as st
import uuid, pickle, faiss, os, glob, re
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from supabase import create_client

def rebuild_file_if_needed(base_name):
    if os.path.exists(base_name):
        try:
            if base_name.endswith(".pkl"):
                with open(base_name,'rb') as f: pickle.load(f)
            if os.path.getsize(base_name) < 1000: raise Exception("corrupted")
            return True
        except:
            try: os.remove(base_name)
            except: pass
    parts = sorted(glob.glob(f"{base_name}.part*"))
    if parts:
        with open(base_name,'wb') as out:
            for p in parts: out.write(open(p,'rb').read())
        return True
    return False

rebuild_file_if_needed("chunks.pkl")
rebuild_file_if_needed("model.faiss")

def get_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
    except:
        url = os.getenv("SUPABASE_URL", "https://yehlwrdkccmbghzngwxg.supabase.co")
        key = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InllaGx3cmRrY2NtYmdoem5nd3hnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk0NzIxNDYsImV4cCI6MjEwNTA0ODE0Nn0.Hrxj0yoa7wGbBP9QUAIC4fwIBQndVTg7WjpSi97uJ7E")
    return create_client(url, key)
supabase=get_supabase()

def get_user_id():
    if st.user.is_logged_in: return st.user.email.replace('@','_at_').replace('.','_')
    return "guest"
def load_chats_for_user(uid):
    if uid=="guest": return None
    try:
        res=supabase.table("chats").select("chats_json").eq("user_id",uid).execute()
        if res.data: return res.data[0]["chats_json"]
    except: pass
    return None
def save_chats_for_user(uid,chats_dict):
    if uid=="guest": return
    try:
        clean={cid:{"title":chat["title"],"messages":[{"role":m["role"],"content":m["content"]} for m in chat["messages"]]} for cid,chat in chats_dict.items()}
        supabase.table("chats").upsert({"user_id":uid,"chats_json":clean}).execute()
    except: pass

st.set_page_config(page_title="Kyle AI",page_icon="🎓",layout="wide")
if "guest_mode" not in st.session_state: st.session_state.guest_mode=False
if not st.user.is_logged_in and not st.session_state.guest_mode:
    st.title("Welcome to Kyle AI 🤖")
    c1,c2=st.columns(2)
    with c1:
        if st.button("Continue with Google",use_container_width=True): st.login()
    with c2:
        if st.button("Continue as Guest",use_container_width=True): st.session_state.guest_mode=True; st.rerun()
    st.stop()

st.markdown("""<style>
.stApp{background:radial-gradient(ellipse at top,#1a2235 0%,#0e1117 70%);}
div[data-testid="stChatMessage"]{background:rgba(30,34,45,0.6)!important;backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.08);border-radius:20px;}
div[data-testid="stChatInput"]>div{background:rgba(30,34,45,0.7)!important;border-radius:24px!important;}
</style>""",unsafe_allow_html=True)

uid=get_user_id()
if "chats" not in st.session_state:
    loaded=load_chats_for_user(uid)
    if loaded: st.session_state.chats=loaded; st.session_state.current_chat=list(loaded.keys())[0]
    else: nid=str(uuid.uuid4())[:8]; st.session_state.current_chat=nid; st.session_state.chats={nid:{"title":"New Chat","messages":[{"role":"assistant","content":"Hey! I'm Kyle AI - AS Level 9618/9709/9702/9231 expert. Ask me anything!"}]}}
    st.session_state.loaded_uid=uid
if st.session_state.get("loaded_uid")!=uid:
    loaded=load_chats_for_user(uid)
    if loaded: st.session_state.chats=loaded; st.session_state.current_chat=list(loaded.keys())[0]
    st.session_state.loaded_uid=uid
if "level" not in st.session_state: st.session_state.level="Best"

TEXT_MODELS=["openai/gpt-oss-20b","openai/gpt-oss-120b","llama-3.3-70b-versatile","llama-3.1-8b-instant","meta-llama/llama-4-maverick-17b-128e-instruct"]
client=Groq(api_key=st.secrets["GROQ_API_KEY"])
def call_groq_auto(messages,max_tokens):
    for mid in TEXT_MODELS:
        try: return client.chat.completions.create(model=mid,messages=messages,max_tokens=max_tokens),mid
        except: continue
    raise Exception("All models failed")

def get_chunks_hash():
    try: return os.path.getmtime("chunks.pkl")
    except: return sum(os.path.getmtime(p) for p in glob.glob("chunks.pkl.part*")) if glob.glob("chunks.pkl.part*") else 0
@st.cache_resource
def load_brain(file_hash):
    rebuild_file_if_needed("chunks.pkl"); rebuild_file_if_needed("model.faiss")
    chunks=pickle.load(open("chunks.pkl","rb")); index=faiss.read_index("model.faiss"); model=SentenceTransformer('all-MiniLM-L6-v2'); return chunks,index,model
chunks,index,embed_model=load_brain(get_chunks_hash())

def fix(t):
    t=t.replace('$$6pt',r'\\[6pt]').replace('$$8pt',r'\\[8pt]').replace('$$12pt',r'\\[12pt]')
    t=t.replace(r'\[','$$').replace(r'\]','$$').replace(r'\(','$').replace(r'\)','$')
    t=t.replace('$$$$','$$').replace('○','- ').replace('•','- ')
    return t
def is_greeting(t): return len(t.strip())<25 and any(g in t.lower() for g in ["hi","hie","hello","hey","thanks","yo","morning","ok"])

def get_system_prompt(level, context, greeting=False, generic=False):
    if greeting: return "You are Kyle AI. Greeting only. Reply friendly short. Do NOT say Subject detected."
    if generic: return "You are Kyle AI - helpful friendly AI like ChatGPT. User question NOT related to Cambridge 9618/9709/9702/9231. Respond normally as general AI. Do NOT say Subject detected."
    instruction={"Simple":"SHORT 3000 tokens. Direct M1 A1 only.","Moderate":"DETAILED 5000 tokens. Explain steps.","Best":"BEST MAX 8192 tokens. Full mark scheme style: start with **Subject detected: XXXX**, state paper/year, M1 A1 B1 FT bold, full explanation, LaTeX $$...$$, common mistakes, boxed final."}[level]
    return f"""You are Kyle AI - Cambridge AS expert.
{instruction}
RULES:
1. AS SYLLABUS ONLY: 9709 AS = P1 + M1/S1 only, 9702 AS = kinematics/dynamics/forces/work/energy/matter/waves/DC/particle, 9618 AS = fundamentals/networks/data representation/programming basics/AS databases, 9231 = AS depth only. If A2 topic say "Outside AS syllabus (A2) but AS foundation is..."
2. LATEX: ALWAYS $$...$$ and $...$. NEVER \\( \\) or \\[ \\]. Never $$6pt. Use \\\\ for newline.
3. If Cambridge: Start **Subject detected: XXXX - Name**. If NOT Cambridge: respond as normal AI, no Subject detected.
CONTEXT (AS only):
{context}
"""

# --- SIDEBAR WITH DELETE ---
with st.sidebar:
    st.markdown("## 🎓 Kyle AI")
    st.caption(f"📚 {len(chunks)} chunks • AS Only")
    if st.user.is_logged_in:
        st.write(f"👤 {st.user.email}")
        if st.button("Logout",use_container_width=True): save_chats_for_user(uid,st.session_state.chats); st.logout()
    else:
        if st.button("Login with Google",use_container_width=True): st.login()
        if st.button("Exit Guest",use_container_width=True): st.session_state.guest_mode=False; st.rerun()

    if st.button("➕ New Chat",use_container_width=True,type="primary"):
        nid=str(uuid.uuid4())[:8]; st.session_state.current_chat=nid; st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"New chat!"}]}; save_chats_for_user(uid,st.session_state.chats); st.rerun()

    if st.button("🗑️ Delete Current Chat",use_container_width=True):
        if len(st.session_state.chats)>1:
            del st.session_state.chats[st.session_state.current_chat]
            st.session_state.current_chat=list(st.session_state.chats.keys())[0]
        else:
            nid=str(uuid.uuid4())[:8]; st.session_state.chats={nid:{"title":"New Chat","messages":[{"role":"assistant","content":"New chat!"}]}}; st.session_state.current_chat=nid
        save_chats_for_user(uid,st.session_state.chats); st.rerun()

    if st.button("🗑️ Delete All Chats",use_container_width=True):
        nid=str(uuid.uuid4())[:8]; st.session_state.chats={nid:{"title":"New Chat","messages":[{"role":"assistant","content":"All chats deleted. New chat!"}]}}; st.session_state.current_chat=nid
        save_chats_for_user(uid,st.session_state.chats); st.rerun()

    st.divider()
    st.caption("Recent Chats (click to open, 🗑️ to delete)")

    # List chats with delete button for each
    for cid in list(st.session_state.chats.keys())[::-1][:20]:
        chat=st.session_state.chats[cid]
        col1,col2=st.columns([0.8,0.2])
        with col1:
            if st.button(chat["title"][:22],key=f"open_{cid}",use_container_width=True):
                st.session_state.current_chat=cid; st.rerun()
        with col2:
            if st.button("🗑️",key=f"del_{cid}",use_container_width=True):
                if len(st.session_state.chats)>1:
                    del st.session_state.chats[cid]
                    if st.session_state.current_chat==cid:
                        st.session_state.current_chat=list(st.session_state.chats.keys())[0]
                else:
                    nid=str(uuid.uuid4())[:8]; st.session_state.chats={nid:{"title":"New Chat","messages":[{"role":"assistant","content":"New chat!"}]}}; st.session_state.current_chat=nid
                save_chats_for_user(uid,st.session_state.chats); st.rerun()

current=st.session_state.chats[st.session_state.current_chat]
st.title("Cambridge AI - AS Level")
st.caption(f"Level: {st.session_state.level} • {len(chunks)} chunks • No image upload")

for m in current["messages"]:
    with st.chat_message(m["role"]): st.markdown(fix(m["content"]))

c1,c2,c3=st.columns(3)
with c1:
    if st.button("Simple 3000",use_container_width=True,type="primary" if st.session_state.level=="Simple" else "secondary"): st.session_state.level="Simple"; st.rerun()
with c2:
    if st.button("Moderate 5000",use_container_width=True,type="primary" if st.session_state.level=="Moderate" else "secondary"): st.session_state.level="Moderate"; st.rerun()
with c3:
    if st.button("Best MAX",use_container_width=True,type="primary" if st.session_state.level=="Best" else "secondary"): st.session_state.level="Best"; st.rerun()

prompt=st.chat_input("Ask any Cambridge question...")

if prompt:
    if current["title"]=="New Chat": current["title"]=prompt[:35]
    current["messages"].append({"role":"user","content":prompt})
    with st.chat_message("assistant"):
        ph=st.empty(); token_map={"Simple":3000,"Moderate":5000,"Best":8192}
        generic_triggers=["joke","story","who are you","what can you do","weather","essay","poem","recipe","movie","game","life advice","relationship","capital of","history of"]
        is_generic_q = any(t in prompt.lower() for t in generic_triggers)

        if is_greeting(prompt):
            sys_prompt=get_system_prompt(st.session_state.level,"",greeting=True)
            msgs=[{"role":"system","content":sys_prompt},{"role":"user","content":prompt}]
            resp,used=call_groq_auto(msgs,600); ans=fix(resp.choices[0].message.content); ph.empty(); st.markdown(ans); current["messages"].append({"role":"assistant","content":ans})
        elif is_generic_q:
            sys_prompt=get_system_prompt(st.session_state.level,"",generic=True)
            msgs=[{"role":"system","content":sys_prompt},{"role":"user","content":prompt}]
            resp,used=call_groq_auto(msgs,1000); ans=fix(resp.choices[0].message.content); ph.empty(); st.markdown(ans); current["messages"].append({"role":"assistant","content":ans})
        else:
            ph.markdown("🔍 Searching AS past papers...")
            q_emb=embed_model.encode([prompt]); D,I=index.search(np.array(q_emb).astype('float32'),25); raw_all=[chunks[i] for i in I[0]]
            def get_y(t): yrs=re.findall(r'(20[1-2][0-9])',t); return max([int(y) for y in yrs]) if yrs else 0
            raw_str_list=[(c["text"] if isinstance(c,dict) else c) for c in raw_all]
            qp=[]; ms=[]
            for txt in raw_str_list:
                if any(a2 in txt.lower() for a2 in ["p3","paper 3","m2","s2","a2 level"]): continue
                if any(k in txt for k in ["M1","A1","B1","M0","FT"]): ms.append(txt)
                else: qp.append(txt)
            qp_s=sorted(qp,key=get_y,reverse=True); ms_s=sorted(ms,key=get_y,reverse=True); best_qp=qp_s[0] if qp_s else (raw_str_list[0] if raw_str_list else "")
            if ms_s: context=f"AS QP:\n{best_qp}\n\nAS MS:\n" + "\n---\n".join(ms_s[:6])
            else: context="\n---\n".join(sorted(raw_str_list,key=get_y,reverse=True)[:6])
            sys_prompt=get_system_prompt(st.session_state.level,context,greeting=False)
            msgs=[{"role":"system","content":sys_prompt},{"role":"user","content":prompt}]
            resp,used=call_groq_auto(msgs,token_map[st.session_state.level]); ans=fix(resp.choices[0].message.content); ph.empty(); st.markdown(ans); current["messages"].append({"role":"assistant","content":ans})
    save_chats_for_user(uid,st.session_state.chats); st.rerun()
