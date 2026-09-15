import streamlit as st
import uuid, pickle, faiss, base64, os, glob, re
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from groq import Groq
from io import BytesIO
from supabase import create_client

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
            clean_msgs = []
            for m in chat["messages"]:
                clean_msgs.append({"role": m["role"], "content": m["content"]})
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

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at top, #1a2235 0%, #0e1117 70%); }
div[data-testid="stChatMessage"] { background: rgba(30,34,45,0.6)!important; backdrop-filter: blur(12px); border:1px solid rgba(255,255,255,0.08); border-radius:20px; }
div[data-testid="stChatInput"] > div { background: rgba(30,34,45,0.7)!important; border-radius:24px!important; border:1px solid rgba(255,255,255,0.1)!important; }
</style>
""", unsafe_allow_html=True)

uid = get_user_id()
if "chats" not in st.session_state:
    loaded = load_chats_for_user(uid)
    if loaded:
        st.session_state.chats=loaded
        st.session_state.current_chat=list(loaded.keys())[0]
    else:
        nid=str(uuid.uuid4())[:8]
        st.session_state.current_chat=nid
        st.session_state.chats={nid:{"title":"New Chat","messages":[{"role":"assistant","content":"Hey! Ask or 📷 add picture with +"}]}}
    st.session_state.loaded_uid=uid

if st.session_state.get("loaded_uid")!=uid:
    loaded=load_chats_for_user(uid)
    if loaded:
        st.session_state.chats=loaded
        st.session_state.current_chat=list(loaded.keys())[0]
    st.session_state.loaded_uid=uid

if "level" not in st.session_state: st.session_state.level="Simple"

# === WORKING MODELS ===
TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
FALLBACK_VISION = "llama-3.2-11b-vision-preview"
FALLBACK_TEXT = "llama-3.1-8b-instant"

client=Groq(api_key=st.secrets["GROQ_API_KEY"])

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
    t = t.replace('○','-').replace('•','- ')
    return t

def get_system_prompt(level, context, is_greeting=False):
    if is_greeting:
        return f"""You are Kyle AI, friendly Cambridge assistant for 9618 CS, 9709 Maths, 9702 Physics, 9231 Further Maths.
User just said hi. Reply friendly like "Hie! I'm Kyle AI 👋 Ready to help with past paper questions? Upload a pic or ask anything."
Do NOT say 'Subject detected'. Do NOT mention mark scheme. Keep short.
"""
    return f"""You are Cambridge AI for 9618 CS, 9709 Maths, 9702 Physics, 9231 Further Maths.

RULE: Only show Subject detected if academic question. For hi/hello/thanks, NEVER show it.

ACADEMIC SUBJECT DETECTION - CHECK IN ORDER:
1. polar, r = f(theta), de Moivre, complex, matrix, eigenvalue, hyperbolic, Maclaurin -> 9231 Further Maths. TOP PRIORITY. polar graphs = ALWAYS 9231.
2. force, velocity, electric field, magnetic, quantum, wave -> 9702 Physics
3. algorithm, database, SQL, network, binary, stack, OOP, logic gates -> 9618 CS
4. ELSE -> 9709 Maths

FLOW:
- I give you SIMILAR QP FOUND + EQUIVALENT MS
- State which paper similar question came from
- Use MS with M1 A1 B1 bold keywords
- Mention year

OUTPUT FOR ACADEMIC ONLY:
- Start with **Subject detected: [CODE] [Name]**
- LaTeX: $$A = \\frac{{1}}{{2}} \\int_{{\\alpha}}^{{\\beta}} [f(\\theta)]^2 d\\theta$$
- Never write (theta), write \\theta

Level: {level}
Simple 3000 short, Moderate 5000 detailed, Best 8192 full

CONTEXT (QP + MS LINKED):
{context}
"""

def to_b64(file):
    try:
        img = Image.open(file)
        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = background
        else:
            img = img.convert("RGB")
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"Image error: {e}")
        return None

def is_greeting_msg(t):
    t = t.lower().strip()
    if len(t) > 30: return False
    greetings = ["hi","hie","hii","hello","hey","how are you","whats up","sup","good morning","good afternoon","good evening","thanks","thank you","okay","ok","yo","morning"]
    return any(g in t for g in greetings)

with st.sidebar:
    st.markdown("## 🎓 Cambridge AI")
    st.caption(f"📚 Loaded {len(chunks)} chunks")
    if st.user.is_logged_in:
        st.write(f"{st.user.email}")
        if st.button("Logout", use_container_width=True): save_chats_for_user(uid, st.session_state.chats); st.logout()
    else:
        if st.button("Login with Google", use_container_width=True): st.login()
        if st.button("Exit Guest", use_container_width=True): st.session_state.guest_mode=False; st.rerun()
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        nid=str(uuid.uuid4())[:8]
        st.session_state.current_chat=nid
        st.session_state.chats[nid]={"title":"New Chat","messages":[{"role":"assistant","content":"Hey!"}]}
        save_chats_for_user(uid, st.session_state.chats); st.rerun()
    st.divider()
    for cid, chat in list(st.session_state.chats.items())[::-1][:15]:
        if st.button(chat["title"][:22], key=f"hist_{cid}", use_container_width=True):
            st.session_state.current_chat=cid; st.rerun()

current=st.session_state.chats[st.session_state.current_chat]
st.title("Cambridge AI")
st.caption(f"{st.session_state.level} • {len(chunks)} chunks • QP->MS Linked • Model: {TEXT_MODEL}")
for m in current["messages"]:
    with st.chat_message(m["role"]):
        if "image_bytes" in m: st.image(m["image_bytes"], width=350)
        st.markdown(fix(m["content"]))

c1,c2,c3=st.columns(3)
with c1:
    if st.button("Simple 3000", use_container_width=True, type="primary" if st.session_state.level=="Simple" else "secondary"):
        st.session_state.level="Simple"; st.rerun()
with c2:
    if st.button("Moderate 5000", use_container_width=True, type="primary" if st.session_state.level=="Moderate" else "secondary"):
        st.session_state.level="Moderate"; st.rerun()
with c3:
    if st.button("Best MAX", use_container_width=True, type="primary" if st.session_state.level=="Best" else "secondary"):
        st.session_state.level="Best"; st.rerun()

prompt_data = st.chat_input("ask...", accept_file=True, file_type=["jpg","jpeg","png","webp"])
if prompt_data:
    text = prompt_data.text if hasattr(prompt_data, 'text') else str(prompt_data)
    files = prompt_data.files if hasattr(prompt_data, 'files') else []
    b64=None; img_bytes=None
    if len(files)>0:
        try:
            img_bytes = files[0].getvalue()
            b64 = to_b64(files[0])
        except:
            img_bytes = None
            b64 = None
        if not text:
            text="Explain this picture using marking scheme, detect subject"

    if text:
        greet = is_greeting_msg(text) and not b64
        if current["title"]=="New Chat": current["title"]=text[:35]
        umsg={"role":"user","content":text}
        if img_bytes: umsg["image_bytes"]=img_bytes
        current["messages"].append(umsg)

        with st.chat_message("assistant"):
            ph=st.empty()
            token_map={"Simple":3000,"Moderate":5000,"Best":8192}

            if greet:
                ph.markdown("👋...")
                system_prompt = get_system_prompt(st.session_state.level, "", is_greeting=True)
                try:
                    resp=client.chat.completions.create(model=TEXT_MODEL,
                        messages=[{"role":"system","content": system_prompt},{"role":"user","content": text}],
                        max_tokens=500)
                except:
                    resp=client.chat.completions.create(model=FALLBACK_TEXT,
                        messages=[{"role":"system","content": system_prompt},{"role":"user","content": text}],
                        max_tokens=500)
                ans=fix(resp.choices[0].message.content)
                ph.empty(); st.markdown(ans)
                current["messages"].append({"role":"assistant","content":ans})
            else:
                ph.markdown("🔍 Step 1: Finding similar Question Paper...")
                q_emb=embed_model.encode([text])
                D,I=index.search(np.array(q_emb).astype('float32'),25)
                raw_all = [chunks[i] for i in I[0]]

                def get_y(t):
                    yrs = re.findall(r'(20[1-2][0-9])', t)
                    return max([int(y) for y in yrs]) if yrs else 0

                raw_str_list = [ (c["text"] if isinstance(c, dict) else c) for c in raw_all ]
                qp_like = []
                ms_like = []
                for txt in raw_str_list:
                    if any(k in txt for k in ["M1","A1","B1","M0","A0","FT","mark scheme"]):
                        ms_like.append(txt)
                    else:
                        qp_like.append(txt)

                qp_like_sorted = sorted(qp_like, key=get_y, reverse=True)
                ms_like_sorted = sorted(ms_like, key=get_y, reverse=True)
                best_qp = qp_like_sorted[0] if qp_like_sorted else raw_str_list[0]

                ph.markdown(f"✅ Similar QP found\n\n🔍 Step 2: Pulling Mark Scheme...")

                if ms_like_sorted:
                    context = f"SIMILAR QUESTION PAPER FOUND:\n{best_qp}\n\n---\n\nEQUIVALENT MARK SCHEME (Latest year first):\n" + "\n\n---\n\n".join(ms_like_sorted[:6])
                else:
                    context = "\n\n---\n\n".join(sorted(raw_str_list, key=get_y, reverse=True)[:7])

                system_prompt = get_system_prompt(st.session_state.level, context, is_greeting=False)

                try:
                    if b64:
                        resp=client.chat.completions.create(model=VISION_MODEL,
                            messages=[{"role":"system","content": system_prompt},
                                      {"role":"user","content":[{"type":"text","text": text},{"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}}]}],
                            max_tokens=token_map[st.session_state.level])
                    else:
                        resp=client.chat.completions.create(model=TEXT_MODEL,
                            messages=[{"role":"system","content": system_prompt},{"role":"user","content": text}],
                            max_tokens=token_map[st.session_state.level])
                except Exception as e:
                    print(f"Primary failed {e}, fallback")
                    if b64:
                        resp=client.chat.completions.create(model=FALLBACK_VISION,
                            messages=[{"role":"system","content": system_prompt},
                                      {"role":"user","content":[{"type":"text","text": text},{"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}}]}],
                            max_tokens=token_map[st.session_state.level])
                    else:
                        resp=client.chat.completions.create(model=FALLBACK_TEXT,
                            messages=[{"role":"system","content": system_prompt},{"role":"user","content": text}],
                            max_tokens=token_map[st.session_state.level])

                ans=fix(resp.choices[0].message.content)
                ph.empty(); st.markdown(ans)
                current["messages"].append({"role":"assistant","content":ans})

        save_chats_for_user(uid, st.session_state.chats)
        st.rerun()
