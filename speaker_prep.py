import json
import os
from pathlib import Path
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # 新增導入
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

SENTENCE_TRANSFORMER_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2' # 使用與 prepro_sub.py 相同的模型

# Configuration
DATA_DIR = Path("data/raw")
OUTPUT_JSON_PATH = Path("data/processed/speaker_data.json")
JSON_FILES = [
    "coscup_2020_session.json",
    "coscup_2021_session.json",
    "coscup_2022_session.json",
    "coscup_2023_session.json",
    "coscup_2024_session.json",
]

# Ensure output directory exists
OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load Sentence Transformer model
print(f"Loading sentence transformer model: {SENTENCE_TRANSFORMER_MODEL}...")
st_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
print("Model loaded.")

# Define Topic Prompts (Copied from prepro_sub.py)
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
    "Design":     ["Software architecture, system design, API design principles, UX research, product design strategy, service design"],
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

# Define mapping from TOPIC_PROMPTS keys to the three main categories
TOPIC_TO_MAIN_CATEGORY_MAP = {
    "Programming": [
        "Programming", "FrontEnd", "BackEnd", "Mobile", "Testing/QA"
    ],
    "Tooling": [
        "Infra", "DevOps", "SaaS", "Hardware/IoT/Maker", "GIS/Mapping",
        "Multimedia/CreativeTech"
    ],
    "Domain": [
        "Data", "Security", "Application", "Community", "StartUp", "Enterprise",
        "Academic", "AI/ML", "Blockchain/Web3", "Design", "CivicTech",
        "FinTech/DeFi", "EdTech", "Telecom/5G", "DevLifestyle",
        "MyData/PersonalData", "Culture/Localization", "Legal/Policy", "OtherTopic"
    ]
}

# Validate that all TOPIC_PROMPTS are covered in the mapping
all_mapped_topics = set()
for category_topics in TOPIC_TO_MAIN_CATEGORY_MAP.values():
    all_mapped_topics.update(category_topics)

for topic_key in TOPIC_PROMPTS.keys():
    if topic_key not in all_mapped_topics:
        print(f"Warning: Topic '{topic_key}' from TOPIC_PROMPTS is not mapped in TOPIC_TO_MAIN_CATEGORY_MAP.")

# Pre-calculate topic embeddings with progress bar
print("Pre-calculating topic embeddings...")
topic_embeddings = {}
for k, prompts in tqdm(TOPIC_PROMPTS.items(), desc="Computing topic embeddings"):
    prompt_embeddings = st_model.encode(prompts, convert_to_numpy=True)
    topic_embeddings[k] = prompt_embeddings.mean(axis=0)

def get_sentence_embedding(text):
    """Get sentence embedding for input text"""
    return st_model.encode([text])[0]

def get_topic_embeddings():
    """Get embeddings for all topic descriptions"""
    topic_embeddings = {}
    for topic, descriptions in TOPIC_PROMPTS.items():
        # Get embedding for the topic description
        topic_emb = st_model.encode(descriptions)[0]
        topic_embeddings[topic] = topic_emb
    return topic_embeddings

def get_topic_tags(text_embedding, top_n=3):
    """
    Get top N most relevant topic tags for a given text embedding
    Args:
        text_embedding: The embedding vector of the input text
        top_n: Number of top topics to return
    Returns:
        List of topic tags ordered by relevance
    """
    # Get topic embeddings (could be cached for better performance)
    topic_embeddings = get_topic_embeddings()
    
    # Calculate similarity scores
    similarities = {}
    for topic, topic_emb in topic_embeddings.items():
        sim = cosine_similarity(
            text_embedding.reshape(1, -1),
            topic_emb.reshape(1, -1)
        )[0][0]
        similarities[topic] = sim
    
    # Sort topics by similarity score and get top N
    sorted_topics = sorted(
        similarities.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [topic for topic, _ in sorted_topics[:top_n]]

def extract_keywords(text, top_n=10):
    """
    Extracts keywords from the given text using TF-IDF.
    """
    # Chinese text segmentation
    seg_list = jieba.cut(text, cut_all=False)
    text_segmented = " ".join(seg_list)
    
    if not text_segmented.strip():
        return [] # Return empty list for empty text

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text_segmented])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Get the top N keywords
        top_indices = tfidf_scores.argsort()[-top_n:][::-1]
        return [feature_names[i] for i in top_indices]
    except ValueError:
        # Handle cases where TF-IDF cannot be computed (e.g., text has only stop words)
        return []

def convert_to_traditional(text):
    # TODO: 可用 opencc 或其他繁簡轉換工具
    return text  # 目前直接回傳原文

def process_speaker_data():
    """處理講者資料，提取標籤"""
    all_speakers = [] 
    all_sessions = []
    speaker_tag_map = {}  
    speaker_sessions = {}  
    
    # 顯示總體進度
    print("\nProcessing speaker data...")
    
    # 1. 讀取檔案
    print("\nPhase 1: Reading JSON files...")
    for json_file in tqdm(JSON_FILES, desc="Loading files"):
        with open(DATA_DIR / json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            year = int(json_file.split("_")[1])
            
            # 處理每個 session
            sessions = data.get("sessions", [])
            for session in tqdm(sessions, desc=f"Processing {year} sessions", leave=False):
                session_id = session.get("id", "N/A")
                title = session.get("zh", {}).get("title", "") or session.get("en", {}).get("title", "")
                desc = session.get("zh", {}).get("description", "") or session.get("en", {}).get("description", "")
                
                # 獲得session的主題標籤
                text = f"{title} {desc}"
                session_embedding = get_sentence_embedding(text)
                session_tags = get_topic_tags(session_embedding, top_n=3)
                
                # 儲存session資訊
                session_obj = {
                    "id": session_id,
                    "title": title,
                    "description": desc,
                    "year": year,
                    "tags": session_tags,
                    "speakers": session.get("speakers", [])
                }
                all_sessions.append(session_obj)
                
                # 更新講者的標籤和演講
                for speaker_id in session.get("speakers", []):
                    speaker_tag_map.setdefault(speaker_id, set()).update(session_tags)
                    speaker_sessions.setdefault(speaker_id, []).append(session_obj)
    
    # 2. 彙整講者資料
    print("\nPhase 2: Aggregating speaker data...")

    def get_speaker_name(speaker_id, all_sessions):
        """
        Retrieve the speaker's name from all_sessions.
        """
        for session in all_sessions:
            for speaker in session.get("speakers", []):
                if isinstance(speaker, dict) and speaker.get("id") == speaker_id:
                    return speaker.get("name", "")
        return ""

    for speaker_id, tags in tqdm(speaker_tag_map.items(), desc="Processing speakers"):
        speaker_obj = {
            "id": speaker_id,
            "name": get_speaker_name(speaker_id, all_sessions),
            "tags": sorted(list(tags)),
            "sessions": speaker_sessions[speaker_id]
        }
        all_speakers.append(speaker_obj)

    # 3. 準備最終輸出
    print("\nPhase 3: Preparing final output...")
    final_output = {
        "speakers": all_speakers,
        "sessions": all_sessions,
        "topic_to_main_category_map": TOPIC_TO_MAIN_CATEGORY_MAP
    }
    
    # 4. 儲存結果
    print("\nPhase 4: Saving results...")
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as outfile:
        json.dump(final_output, outfile, ensure_ascii=False, indent=4)
    print(f"Successfully generated {OUTPUT_JSON_PATH}")

    return final_output

if __name__ == "__main__":
    process_speaker_data()