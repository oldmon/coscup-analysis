import os, pathlib, glob, json, time, numpy as np, pandas as pd
import openai
from tqdm import tqdm


KEY_FILE = pathlib.Path("key.txt") 

MODEL      = "text-embedding-3-small"  
BATCH_SIZE = 100                      

# 關鍵字塞這裡
TOPIC_PROMPTS = {
    "OtherTopic": ["General technology talk"],
    "Data":       ["Big data analytics and databases"],
    "Security":   ["Information security and cybersecurity"],
    "Application":["Software application development"],
    "Infra":      ["Cloud infrastructure and DevOps"],
    "FrontEnd":   ["Web front-end development and UI/UX"],
    "BackEnd":    ["API backend and microservices"],
    "SaaS":       ["Software-as-a-Service business model"],
    "Community":  ["Open-source community and collaboration"],
    "StartUp":    ["Startup innovation and entrepreneurship"],
    "Enterprise": ["Large-scale enterprise solutions"],
    "Academic":   ["Academic research and education"]
}

rows, texts, years = [], [], []
for fp in sorted(glob.glob("coscup_score_20*.json")):
    data = json.load(open(fp, encoding="utf-8"))
    yc = {}
    for d in data: yc[d["year"]] = yc.get(d["year"], 0) + 1

    for d in data:
        txt = f'{d["title"]} {d["description"]}'.replace("\n", " ")
        rows.append({"SSID": d["id"], "Year": d["year"]})
        texts.append(txt)
        years.append(d["year"])

def embed(texts):
    out, i = [], 0
    pbar = tqdm(total=len(texts), desc="Embedding sessions")
    while i < len(texts):
        chunk = texts[i:i+BATCH_SIZE]
        try:
            res = openai.embeddings.create(
                model=MODEL,
                input=chunk
            )
            out.extend([e.embedding for e in res.data])
            i += len(chunk)
            pbar.update(len(chunk))
        except openai.RateLimitError:
            time.sleep(10)
    pbar.close()
    return np.array(out, dtype=np.float32)

session_vecs = embed(texts)

topic_vecs = {}
for k, prompts in TOPIC_PROMPTS.items():
    resp = openai.embeddings.create(model=MODEL, input=prompts)
    vecs = np.array([e.embedding for e in resp.data], dtype=np.float32)
    topic_vecs[k] = vecs.mean(axis=0)

# 0-100 
def cos(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = {k: [] for k in TOPIC_PROMPTS}
for v in tqdm(session_vecs, desc="Scoring"):
    for k, tv in topic_vecs.items():
        s = cos(v, tv)
        score = int(round((s + 1) / 2 * 100))  
        scores[k].append(score)


df = pd.DataFrame(rows)
for k in TOPIC_PROMPTS:
    df[k] = scores[k]
cnt = pd.Series(years).value_counts().to_dict()
df["SessionCount"] = df["Year"].map(cnt)
COLS = ["SSID", "Year"] + list(TOPIC_PROMPTS) + ["SessionCount"]

df[COLS].to_csv("coscup_score.csv", index=False, encoding="utf-8")

