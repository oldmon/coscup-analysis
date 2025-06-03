import json
import os
from pathlib import Path
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # 新增導入
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

# Pre-calculate topic embeddings
topic_embeddings = {}
for k, prompts in TOPIC_PROMPTS.items():
    prompt_embeddings = st_model.encode(prompts, convert_to_numpy=True)
    topic_embeddings[k] = prompt_embeddings.mean(axis=0) # Average embeddings for multiple prompts per topic

def get_sentence_embedding(text):
    """
    Generates a sentence embedding for the given text using SentenceTransformer.
    """
    # Ensure text is not empty, as encode might behave unexpectedly
    if not text.strip():
        return np.zeros(st_model.get_sentence_embedding_dimension()).tolist() # Return zero vector for empty text
    return st_model.encode(text).tolist()

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

def process_speaker_data():
    """
    Processes speaker data from JSON files, extracts embeddings and keywords,
    and saves the processed data to a JSON file.
    """
    all_speakers = []

    for json_file in JSON_FILES:
        file_path = DATA_DIR / json_file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        speakers = data.get("speakers", [])
        sessions = data.get("sessions", [])

        for speaker in speakers:
            speaker_id = speaker["id"]
            # Collect all session titles and descriptions for the speaker
            speaker_text = speaker.get("zh", {}).get("bio", "") + "\n".join(
                [
                    session.get("zh", {}).get("title", "") + " " + session.get("zh", {}).get("description", "")
                    for session in sessions
                    if speaker_id in session.get("speakers", [])
                ]
            )

            # Generate speaker embedding
            speaker_embedding = get_sentence_embedding(speaker_text)

            # Calculate topic scores
            topic_scores = {}
            if speaker_embedding and any(speaker_embedding): # Only calculate if embedding is not zero vector
                 # Reshape for cosine_similarity: (n_samples, n_features)
                speaker_embedding_reshaped = np.array(speaker_embedding).reshape(1, -1)
                for topic, topic_emb in topic_embeddings.items():
                    topic_emb_reshaped = topic_emb.reshape(1, -1)
                    score = cosine_similarity(speaker_embedding_reshaped, topic_emb_reshaped)[0][0]
                    # Optional: Scale score to 0-100 or similar range if needed for visualization
                    topic_scores[topic] = float(score) # Convert numpy float to standard float for JSON

            # Extract keywords (optional, but can still be useful)
            keywords = extract_keywords(speaker_text)

            all_speakers.append(
                {
                    "id": speaker_id,
                    "name": speaker.get("zh", {}).get("name", speaker.get("en", {}).get("name", "N/A")), # Fallback to English name
                    "topic_scores": topic_scores,
                    "keywords": keywords # Include extracted keywords
                }
            )

    # Prepare the final output structure
    final_output = {
        "speakers": all_speakers,
        "topic_to_main_category_map": TOPIC_TO_MAIN_CATEGORY_MAP
    }

    # Save the processed data to a JSON file
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as outfile:
        json.dump(final_output, outfile, ensure_ascii=False, indent=4)

    print(f"Successfully generated {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    process_speaker_data()