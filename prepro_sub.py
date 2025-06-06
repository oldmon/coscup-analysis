import os, pathlib, glob, json, time, numpy as np, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

SENTENCE_TRANSFORMER_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
print(f"Loading sentence transformer model: {SENTENCE_TRANSFORMER_MODEL}...")
st_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
print("Model loaded.")


# 關鍵字塞這裡
TOPIC_PROMPTS = {
    # Refined Original 12 Topics
    "OtherTopic": ["General technology discussions, miscellaneous topics, and talks not fitting other specific categories"],
    "Data":       ["Data science, big data, data analytics, databases (SQL, NoSQL), data engineering, data visualization, business intelligence"],
    "Security":   ["Information security, cybersecurity, network security, application security, cryptography, privacy-enhancing technologies, ethical hacking"],
    "Application":["Software application development, desktop applications, mobile-adjacent applications, user-facing tools and utilities"],
    "Infra":      ["Cloud infrastructure (public/private/hybrid), server management, system administration, networking infrastructure, data centers"],
    "FrontEnd":   ["Web front-end development, UI/UX design for web, client-side JavaScript frameworks (React, Vue, Angular), HTML, CSS, web performance"],
    "BackEnd":    ["Server-side development, API design and development, microservices architecture, backend frameworks, serverless functions for backend logic"],
    "SaaS":       ["Software-as-a-Service business models, cloud-based product delivery, subscription services, multi-tenancy architecture"],
    "Community":  ["Open-source community building, collaboration, contribution strategies, diversity and inclusion in tech, open source project governance"],
    "StartUp":    ["Startup ecosystem, entrepreneurship, product development for startups, venture capital, innovation, lean startup methodologies"],
    "Enterprise": ["Enterprise software solutions, large-scale systems, corporate IT strategy, digital transformation in enterprises, legacy system modernization"],
    "Academic":   ["Academic research in computer science, university programs, scientific computing, research methodologies in tech"],

    # Newly Recommended 18 Topics
    "AI/ML":      ["Artificial intelligence, machine learning, deep learning, natural language processing (NLP), computer vision, AGI, AI ethics and responsible AI"],
    "Mobile":     ["Mobile application development (iOS, Android), cross-platform frameworks (Flutter, React Native), mobile UI/UX, mobile backend services (MBaaS)"],
    "DevOps":     ["DevOps culture and practices, CI/CD pipelines, containerization (Docker), orchestration (Kubernetes), infrastructure as code (IaC), MLOps, Site Reliability Engineering (SRE)"],
    "Programming":["Specific programming languages (Python, Go, Rust, Java, C++, Julia, Swift, Kotlin), software engineering principles, coding best practices, algorithms, data structures, compilers, interpreters"],
    "Blockchain/Web3": ["Blockchain technology fundamentals, distributed ledger technology (DLT), cryptocurrencies (Bitcoin, Ethereum), smart contracts, decentralized applications (DApps), Web3 ecosystem"],
    "Testing/QA": ["Software testing methodologies, quality assurance processes, test automation frameworks, unit testing, integration testing, performance testing, debugging techniques"],
    "Design":     ["Software architecture, system design, API design principles, design patterns, UX research, product design strategy, service design"],
    "Hardware/IoT/Maker": ["Open source hardware, Internet of Things (IoT) platforms and applications, embedded systems, robotics, maker movement, DIY electronics, 3D printing, FPGAs, RISC-V architecture"],
    "GIS/Mapping":["Geographic Information Systems (GIS), OpenStreetMap (OSM), map rendering technologies, spatial data analysis, location-based services, cartography"],
    "CivicTech":  ["Civic technology, open government initiatives, tech for social good, hacktivism for public benefit, e-participation, digital democracy tools"],
    "FinTech/DeFi": ["Financial technology innovations, Decentralized Finance (DeFi), digital payments, InsurTech, RegTech, blockchain applications in finance"],
    "EdTech":     ["Educational technology, open source tools for education, e-learning platforms, online course development, learning analytics, gamification in education"],
    "Telecom/5G": ["Telecommunications industry, 5G network technology, network function virtualization (NFV), software-defined networking (SDN) in telecom, open RAN, mobile network evolution"],
    "DevLifestyle": ["Developer ergonomics, productivity hacks and tools, career development for engineers, remote work best practices, mental health and well-being in the tech industry"],
    "Multimedia/CreativeTech": ["Audio and video processing, streaming media technologies, creative coding, digital art tools, game development technologies, virtual and augmented reality (VR/AR)"],
    "MyData/PersonalData": ["MyData principles, personal data stores (PDS), data sovereignty, human-centric data management, digital identity solutions, privacy by design for personal data"],
    "Culture/Localization": ["Technology for language revitalization, preservation of cultural heritage through tech, software localization and internationalization, indigenous technology projects"],
    "Legal/Policy": ["Open source licensing (GPL, MIT, Apache), copyright and intellectual property in software, data privacy regulations (GDPR, CCPA), technology policy, legal aspects of AI and emerging tech"]
}

rows, texts = [], []
source_file_pattern = "data/raw/coscup_20*_session.json"
print(f"Looking for session files in: {source_file_pattern}")

json_files_found = sorted(glob.glob(source_file_pattern))
if not json_files_found:
    print(f"No files found matching pattern: {source_file_pattern}")
    print("Please ensure the path and pattern are correct and files exist.")
    exit(1)

for fp_str in sorted(glob.glob(source_file_pattern)):
    fp = pathlib.Path(fp_str)
    year_str = fp.stem.split("_")[1] # Extracts year like "2020" from "coscup_2020_session"
    print(f"Processing file: {fp_str} for year: {year_str}")
    with open(fp, "r", encoding="utf-8") as f:
        data_content = json.load(f)
    
    if "sessions" in data_content:
        for d in data_content["sessions"]:
            zh_title = d.get("zh", {}).get("title", "")
            zh_description = d.get("zh", {}).get("description", "")
            en_title = d.get("en", {}).get("title", "")
            en_description = d.get("en", {}).get("description", "")

            current_title = ""
            current_description = ""

            if zh_title or zh_description: # Prefer Chinese if any part exists
                current_title = zh_title
                current_description = zh_description
            elif en_title or en_description: # Fallback to English if any part exists
                current_title = en_title
                current_description = en_description

            txt = f'{current_title} {current_description}'.replace("\n", " ").strip()
            
            if not txt:
                # print(f"Skipping session {d.get('id', 'N/A')} from year {year_str} due to empty title/description.")
                continue

            rows.append({"SSID": d.get("id", "N/A"), "Year": year_str})
            texts.append(txt)
    else:
        print(f"Warning: 'sessions' key not found in {fp_str}")

print(f"Found {len(texts)} sessions to embed.")
session_vecs = st_model.encode(texts, show_progress_bar=True)

topic_vecs = {}
for k, prompts in tqdm(TOPIC_PROMPTS.items(), desc="Embedding topics"):
    prompt_embeddings = st_model.encode(prompts, show_progress_bar=False)
    vecs = np.array(prompt_embeddings, dtype=np.float32)
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
cnt = df["Year"].value_counts().to_dict()
df["SessionCount"] = df["Year"].map(cnt)
COLS = ["SSID", "Year"] + list(TOPIC_PROMPTS) + ["SessionCount"]

output_csv_path = pathlib.Path("coscup_score_st.csv") # Changed output filename to avoid overwriting
df[COLS].to_csv(output_csv_path, index=False, encoding="utf-8")
print(f"Output saved to {output_csv_path}")
