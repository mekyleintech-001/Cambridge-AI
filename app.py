!pip install -q "gradio==5.23.3" groq PyMuPDF sentence-transformers faiss-cpu > /dev/null

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import faiss, pickle, numpy as np, traceback
from groq import Groq
from sentence_transformers import SentenceTransformer
import gradio as gr

folder = "/content/drive/MyDrive/CambridgeAI"
API_KEY = open(f"{folder}/groq_key.txt").read().strip()
client = Groq(api_key=API_KEY)

# AUTO-FIND WORKING MODEL
try:
    models = client.models.list()
    available = [m.id for m in models.data]
    print("Available:", available[:10])
    for pref in ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]:
        if pref in available:
            WORKING_MODEL = pref
            break
    else:
        WORKING_MODEL = available[0]
    print("Using model:", WORKING_MODEL)
except Exception as e:
    print("List failed, using fallback:", e)
    WORKING_MODEL = "llama-3.3-70b-versatile"

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(f"{folder}/model.faiss")
chunks = pickle.load(open(f"{folder}/chunks.pkl","rb"))
print(f"Ready: {len(chunks)}")

def answer_fn(question, style):
    try:
        q_emb = model.encode([question])
        D,I = index.search(np.array(q_emb), k=5)
        context = "\n".join([chunks[i] for i in I[0]])[:4000]
        r = client.chat.completions.create(
            model=WORKING_MODEL,
            messages=[{"role":"system","content":f"Cambridge 9618 tutor, style:{style}. Context:{context}"},{"role":"user","content":question}],
            max_tokens=800
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

with gr.Blocks() as demo:
    gr.Markdown(f"## 🎓 CambridgeAI 9618 | Model: {WORKING_MODEL}")
    style = gr.Radio(["Explain","Exam Question","Mark My Answer"], value="Explain", label="Style")
    chatbot = gr.Chatbot(height=450, type="messages")
    q = gr.Textbox(label="Your question")
    btn = gr.Button("Ask", variant="primary")
    def chat_fn(q_text, s, hist):
        if not q_text.strip(): return hist, ""
        ans = answer_fn(q_text, s)
        hist.append({"role":"user","content":q_text})
        hist.append({"role":"assistant","content":ans})
        return hist, ""
    btn.click(chat_fn, [q, style, chatbot], [chatbot, q])
    q.submit(chat_fn, [q, style, chatbot], [chatbot, q])

demo.launch(share=True)
