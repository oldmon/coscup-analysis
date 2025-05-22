import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import networkx as nx
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models import Word2Vec
from wordcloud import WordCloud
import json
import re
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.colors as colors
import os

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 語意分析與關鍵詞提取
def semantic_analysis(sessions_df, speakers_df):
    """對議程內容和講者背景進行語意分析"""
    print("進行語意分析與關鍵詞提取...")
    
    # 1.1 添加技術關鍵詞詞典
    tech_keywords = [
        'AI', '人工智慧', '機器學習', 'Machine Learning', '深度學習', 'Deep Learning',
        'Python', 'Java', 'JavaScript', 'C++', 'Go', 'Rust', 'PHP', 'Ruby',
        'Linux', 'Windows', 'macOS', 'Unix', 'Android', 'iOS',
        '資料庫', 'Database', 'SQL', 'NoSQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis',
        '雲端', 'Cloud', 'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'K8s', 'Container',
        '安全', 'Security', '加密', 'Encryption', '隱私', 'Privacy',
        'DevOps', 'CI/CD', 'Git', 'GitHub', 'GitLab', 
        '前端', 'Frontend', 'React', 'Vue', 'Angular', '後端', 'Backend', 'Node.js',
        '區塊鏈', 'Blockchain', '智能合約', 'Smart Contract', '以太坊', 'Ethereum',
        '開源', 'Open Source', '社群', 'Community', '貢獻', 'Contribution',
        '資料科學', 'Data Science', '大數據', 'Big Data', '資料分析', 'Data Analysis',
        'IoT', '物聯網', 'Internet of Things', '邊緣運算', 'Edge Computing',
        '網路', 'Network', 'HTTP', 'TCP/IP', 'DNS', 'CDN', 'API', 'RESTful', 'GraphQL'
    ]
    
    for keyword in tech_keywords:
        jieba.add_word(keyword)
    
    # 1.2 分析議程標題和描述
    if 'title' in sessions_df.columns:
        # 載入停用詞
        stopwords = set(['的', '了', '和', '是', '就', '都', '而', '及', '與', '著',
                        '或', '一個', '我們', '你們', '他們', '可以', '這個', '那個'])
        
        # 合併標題和描述
        sessions_df['text'] = sessions_df['title'] + ' ' + sessions_df['description'].fillna('')
        
        # 分詞
        sessions_df['tokens'] = sessions_df['text'].apply(
            lambda x: [w for w in jieba.cut(x) if w not in stopwords and len(w) > 1]
        )
        
        # 使用TF-IDF提取關鍵詞
        all_texts = sessions_df['text'].tolist()
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(all_texts)
        feature_names = tfidf.get_feature_names_out()
        
        # 為每個議程提取關鍵詞
        sessions_df['keywords'] = sessions_df['text'].apply(
            lambda x: jieba.analyse.extract_tags(x, topK=8)
        )
        
        # 繪製整體關鍵詞雲
        all_keywords = []
        for keywords in sessions_df['keywords']:
            all_keywords.extend(keywords)
        
        keyword_counts = Counter(all_keywords)
        
        wordcloud = WordCloud(
            width=800, height=600,
            background_color='white',
            max_words=100,
            font_path='simhei.ttf'  # 需確保有合適的中文字體
        )
        wordcloud.generate_from_frequencies(keyword_counts)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('COSCUP 議程關鍵詞雲')
        plt.savefig('visualizations/static/session_wordcloud.png', dpi=300)
        plt.close()
    
    # 1.3 分析講者背景
    if not speakers_df.empty and 'bio' in speakers_df.columns:
        # 分詞和提取關鍵詞
        speakers_df['bio_tokens'] = speakers_df['bio'].fillna('').apply(
            lambda x: [w for w in jieba.cut(x) if w not in stopwords and len(w) > 1]
        )
        
        speakers_df['bio_keywords'] = speakers_df['bio'].fillna('').apply(
            lambda x: jieba.analyse.extract_tags(x, topK=5)
        )
        
        # 繪製講者背景關鍵詞雲
        all_bio_keywords = []
        for keywords in speakers_df['bio_keywords']:
            all_bio_keywords.extend(keywords)
        
        bio_keyword_counts = Counter(all_bio_keywords)
        
        wordcloud = WordCloud(
            width=800, height=600,
            background_color='white',
            max_words=100,
            font_path='simhei.ttf'
        )
        wordcloud.generate_from_frequencies(bio_keyword_counts)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('COSCUP 講者背景關鍵詞雲')
        plt.savefig('visualizations/static/speaker_bio_wordcloud.png', dpi=300)
        plt.close()
    
    # 1.4 議程相似度分析
    if 'tokens' in sessions_df.columns:
        # 建立文檔-詞項矩陣
        docs = [' '.join(tokens) for tokens in sessions_df['tokens']]
        vectorizer = TfidfVectorizer(max_features=500)
        X = vectorizer.fit_transform(docs)
        
        # 計算議程之間的相似度
        similarity_matrix = cosine_similarity(X)
        
        # 找出每個議程最相似的其他議程
        similar_sessions = []
        for i in range(len(sessions_df)):
            sim_scores = list(enumerate(similarity_matrix[i]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # 排除自己
            sim_scores = sim_scores[1:6]  # 取前5個最相似的
            
            similar_indices = [idx for idx, score in sim_scores if score > 0.3]  # 設置一個閾值
            if similar_indices:
                similar_sessions.append({
                    'session_id': sessions_df.iloc[i]['id'],
                    'title': sessions_df.iloc[i]['title'],
                    'year': sessions_df.iloc[i]['year'],
                    'similar_sessions': [
                        {
                            'session_id': sessions_df.iloc[idx]['id'],
                            'title': sessions_df.iloc[idx]['title'],
                            'year': sessions_df.iloc[idx]['year'],
                            'similarity': float(score)
                        }
                        for idx, score in sim_scores if idx in similar_indices
                    ]
                })
        
        # 儲存相似議程資料
        with open('data/processed/similar_sessions.json', 'w', encoding='utf-8') as f:
            json.dump(similar_sessions, f, ensure_ascii=False, indent=2)
    
    # 1.5 技術關鍵詞趨勢分析
    if 'tokens' in sessions_df.columns and 'year' in sessions_df.columns:
        # 定義要追蹤的技術關鍵詞
        tech_terms = [
            'AI', '人工智慧', '機器學習', 'Machine Learning', '深度學習', 'Deep Learning',
            '資料科學', 'Data Science', '大數據', 'Big Data',
            '雲端', 'Cloud', 'AWS', 'Azure', 'GCP',
            'Docker', 'Kubernetes', 'K8s', 'Container',
            '安全', 'Security', '加密', 'Encryption', '隱私', 'Privacy',
            'DevOps', 'CI/CD',
            '區塊鏈', 'Blockchain',
            'IoT', '物聯網'
        ]
        
        # 追蹤每年這些關鍵詞的出現次數
        years = sorted(sessions_df['year'].unique())
        tech_trends = {term: [] for term in tech_terms}
        
        for year in years:
            year_docs = sessions_df[sessions_df['year'] == year]
            year_tokens = []
            for tokens in year_docs['tokens']:
                year_tokens.extend(tokens)
            
            token_counter = Counter(year_tokens)
            total_tokens = sum(token_counter.values())
            
            # 計算每個技術詞的比例
            for term in tech_terms:
                count = token_counter.get(term, 0)
                # 計算千分比
                ratio = (count / total_tokens) * 1000 if total_tokens > 0 else 0
                tech_trends[term].append(ratio)
        
        # 將結果保存為DataFrame
        tech_trends_df = pd.DataFrame(tech_trends, index=years)
        tech_trends_df.to_csv('data/processed/tech_trends.csv', encoding='utf-8')
        
        # 繪製技術趨勢圖
        plt.figure(figsize=(14, 8))
        
        # 選擇前10個有變化的關鍵詞
        variance = tech_trends_df.var(axis=0).sort_values(ascending=False)
        top_terms = variance.index[:10]
        
        for term in top_terms:
            plt.plot(years, tech_trends_df[term], marker='o', label=term)
        
        plt.title('技術關鍵詞趨勢分析 (2020-2024)')
        plt.xlabel('年份')
        plt.ylabel('出現頻率 (千分比)')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('visualizations/static/tech_trends.png', dpi=300)
        plt.close()
    
    return {
        'session_keywords': sessions_df['keywords'].tolist() if 'keywords' in sessions_df.columns else [],
        'tech_trends_df': tech_trends_df if 'tech_trends_df' in locals() else None,
        'similar_sessions': similar_sessions if 'similar_sessions' in locals() else None
    }

# 2. 講者與議程的語意關係分析
def speaker_session_semantic_analysis(sessions_df, speakers_df):
    """分析講者與議程之間的語意關係"""
    print("分析講者與議程的語意關係...")
    
    # 建立講者到議程的映射
    speaker_to_sessions = defaultdict(list)
    for _, row in sessions_df.iterrows():
        # 假設speakers欄位是講者ID的列表或字符串
        if isinstance(row['speakers'], list):
            speakers = row['speakers']
        elif isinstance(row['speakers'], str):
            try:
                speakers = json.loads(row['speakers'])
                if not isinstance(speakers, list):
                    speakers = row['speakers'].split(',')
            except:
                speakers = row['speakers'].split(',')
                
        speakers = [s.strip() for s in speakers if s.strip()]
        
        for speaker_id in speakers:
            speaker_to_sessions[speaker_id].append(row['id'])
    
    # 2.1 講者專長領域分析
    speaker_expertise = {}
    
    if 'keywords' in sessions_df.columns:
        for speaker_id, session_ids in speaker_to_sessions.items():
            # 取得該講者的所有議程關鍵詞
            speaker_sessions = sessions_df[sessions_df['id'].isin(session_ids)]
            all_keywords = []
            for keywords in speaker_sessions['keywords']:
                all_keywords.extend(keywords)
            
            # 計算關鍵詞頻率
            keyword_counter = Counter(all_keywords)
            
            # 取前10個關鍵詞作為專長領域
            top_keywords = keyword_counter.most_common(10)
            
            speaker_expertise[speaker_id] = {
                'top_keywords': dict(top_keywords),
                'session_count': len(session_ids),
                'years': sorted(speaker_sessions['year'].unique().tolist())
            }
        
        # 合併講者資訊
        if not speakers_df.empty:
            for speaker_id in speaker_expertise:
                speaker_info = speakers_df[speakers_df['id'] == speaker_id]
                if not speaker_info.empty:
                    if 'name' in speaker_info.columns:
                        speaker_expertise[speaker_id]['name'] = speaker_info.iloc[0]['name']
                    if 'company' in speaker_info.columns:
                        speaker_expertise[speaker_id]['company'] = speaker_info.iloc[0]['company']
        
        # 儲存講者專長資料
        with open('data/processed/speaker_expertise.json', 'w', encoding='utf-8') as f:
            json.dump(speaker_expertise, f, ensure_ascii=False, indent=2)
    
    # 2.2 共同主題講者群組分析
    if 'keywords' in sessions_df.columns:
        # 根據關鍵詞將講者分組
        keyword_to_speakers = defaultdict(set)
        
        for speaker_id, info in speaker_expertise.items():
            for keyword in info['top_keywords']:
                keyword_to_speakers[keyword].add(speaker_id)
        
        # 取前20個最多講者的關鍵詞
        popular_keywords = sorted(
            keyword_to_speakers.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:20]
        
        # 繪製每個關鍵詞的講者數量
        plt.figure(figsize=(14, 8))
        keywords = [kw for kw, _ in popular_keywords]
        speaker_counts = [len(speakers) for _, speakers in popular_keywords]
        
        plt.bar(keywords, speaker_counts, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.title('熱門技術關鍵詞講者數量')
        plt.xlabel('關鍵詞')
        plt.ylabel('講者數量')
        plt.tight_layout()
        plt.savefig('visualizations/static/keyword_speaker_counts.png', dpi=300)
        plt.close()
        
        # 儲存關鍵詞-講者映射
        keyword_speakers = {
            keyword: list(speakers) 
            for keyword, speakers in keyword_to_speakers.items()
            if len(speakers) >= 3  # 至少3位講者
        }
        
        with open('data/processed/keyword_speakers.json', 'w', encoding='utf-8') as f:
            json.dump(keyword_speakers, f, ensure_ascii=False, indent=2)
    
    # 2.3 講者主題演變分析
    recurring_speakers = []
    
    for speaker_id, info in speaker_expertise.items():
        # 找出參與多年的講者
        if len(info.get('years', [])) > 1:
            recurring_speakers.append(speaker_id)
    
    if recurring_speakers and 'keywords' in sessions_df.columns:
        # 分析每位重複出席講者的主題變化
        speaker_evolution = {}
        
        for speaker_id in recurring_speakers:
            speaker_sessions = {}
            
            # 獲取該講者每年的議程
            for session_id in speaker_to_sessions[speaker_id]:
                session = sessions_df[sessions_df['id'] == session_id].iloc[0]
                year = session['year']
                
                if year not in speaker_sessions:
                    speaker_sessions[year] = []
                
                speaker_sessions[year].append({
                    'id': session['id'],
                    'title': session['title'],
                    'keywords': session['keywords']
                })
            
            # 計算每年的關鍵詞
            yearly_keywords = {}
            for year, sessions in speaker_sessions.items():
                all_keywords = []
                for session in sessions:
                    all_keywords.extend(session['keywords'])
                
                keyword_counter = Counter(all_keywords)
                yearly_keywords[year] = dict(keyword_counter.most_common(5))
            
            # 添加講者名稱
            name = speaker_expertise[speaker_id].get('name', speaker_id)
            
            speaker_evolution[speaker_id] = {
                'name': name,
                'yearly_keywords': yearly_keywords,
                'years': sorted(yearly_keywords.keys())
            }
        
        # 儲存講者主題演變資料
        with open('data/processed/speaker_evolution.json', 'w', encoding='utf-8') as f:
            json.dump(speaker_evolution, f, ensure_ascii=False, indent=2)
        
        # 繪製部分講者的關鍵詞變化
        # 選擇前5位參與年份最多的講者
        top_speakers = sorted(
            speaker_evolution.items(),
            key=lambda x: len(x[1]['years']),
            reverse=True
        )[:5]
        
        for speaker_id, info in top_speakers:
            plt.figure(figsize=(12, 6))
            
            years = info['years']
            yearly_keywords = info['yearly_keywords']
            
            # 找出所有出現的關鍵詞
            all_kw = set()
            for year, keywords in yearly_keywords.items():
                all_kw.update(keywords.keys())
            
            # 選擇出現頻率較高的關鍵詞
            top_kw = []
            for kw in all_kw:
                total = sum(yearly_keywords.get(year, {}).get(kw, 0) for year in years)
                if total >= 2:  # 至少出現2次
                    top_kw.append(kw)
            
            # 繪製關鍵詞變化趨勢
            for kw in top_kw[:8]:  # 最多顯示8個關鍵詞
                values = [yearly_keywords.get(year, {}).get(kw, 0) for year in years]
                plt.plot(years, values, marker='o', label=kw)
            
            plt.title(f"講者 '{info['name']}' 主題演變 (2020-2024)")
            plt.xlabel('年份')
            plt.ylabel('關鍵詞出現次數')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"visualizations/static/speaker_evolution_{speaker_id}.png", dpi=300)
            plt.close()
    
    return {
        'speaker_expertise': speaker_expertise,
        'keyword_speakers': keyword_speakers if 'keyword_speakers' in locals() else None,
        'speaker_evolution': speaker_evolution if 'speaker_evolution' in locals() else None
    }

# 3. 議程軌(Track)語意分析
def track_semantic_analysis(sessions_df):
    """分析議程軌的語意內容和特點"""
    print("進行議程軌語意分析...")
    
    if 'track' not in sessions_df.columns:
        print("警告: sessions_df 中沒有 track 欄位")
        return {}
    
    # 3.1 分析每個議程軌的關鍵詞特徵
    track_profiles = {}
    
    # 獲取所有議程軌
    tracks = sessions_df['track'].dropna().unique()
    
    for track in tracks:
        # 獲取該議程軌的所有議程
        track_sessions = sessions_df[sessions_df['track'] == track]
        
        # 計算該議程軌的基本統計資料
        years = sorted(track_sessions['year'].unique())
        session_counts = track_sessions.groupby('year').size().to_dict()
        
        # 提取關鍵詞 (如果有)
        if 'keywords' in track_sessions.columns:
            all_keywords = []
            for keywords in track_sessions['keywords']:
                all_keywords.extend(keywords)
            
            keyword_counter = Counter(all_keywords)
            top_keywords = dict(keyword_counter.most_common(15))
        else:
            top_keywords = {}
        
        # 統計講者 (假設有speakers欄位)
        if 'speakers' in track_sessions.columns:
            all_speakers = []
            
            for _, row in track_sessions.iterrows():
                if isinstance(row['speakers'], list):
                    speakers = row['speakers']
                elif isinstance(row['speakers'], str):
                    try:
                        speakers = json.loads(row['speakers'])
                        if not isinstance(speakers, list):
                            speakers = row['speakers'].split(',')
                    except:
                        speakers = row['speakers'].split(',')
                        
                all_speakers.extend([s.strip() for s in speakers if s.strip()])
            
            speaker_counter = Counter(all_speakers)
            top_speakers = dict(speaker_counter.most_common(10))
        else:
            top_speakers = {}
        
        track_profiles[track] = {
            'years': years,
            'session_counts': session_counts,
            'top_keywords': top_keywords,
            'top_speakers': top_speakers,
            'total_sessions': len(track_sessions)
        }
    
    # 儲存議程軌分析結果
    with open('data/processed/track_profiles.json', 'w', encoding='utf-8') as f:
        json.dump(track_profiles, f, ensure_ascii=False, indent=2)
    
    # 3.2 議程軌之間的相似度分析
    if 'keywords' in sessions_df.columns:
        # 為每個議程軌創建關鍵詞向量
        track_vectors = {}
        all_keywords = set()
        
        # 收集所有關鍵詞
        for track, profile in track_profiles.items():
            all_keywords.update(profile['top_keywords'].keys())
        
        # 建立關鍵詞列表
        keyword_list = sorted(all_keywords)
        
        # 為每個議程軌創建向量
        for track, profile in track_profiles.items():
            vector = np.zeros(len(keyword_list))
            
            for i, keyword in enumerate(keyword_list):
                vector[i] = profile['top_keywords'].get(keyword, 0)
            
            # 標準化
            if np.sum(vector) > 0:
                vector = vector / np.sum(vector)
            
            track_vectors[track] = vector
        
        # 計算相似度矩陣
        tracks = sorted(track_vectors.keys())
        similarity_matrix = np.zeros((len(tracks), len(tracks)))
        
        for i, track1 in enumerate(tracks):
            for j, track2 in enumerate(tracks):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    vec1 = track_vectors[track1]
                    vec2 = track_vectors[track2]
                    # 計算餘弦相似度
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) if np.linalg.norm(vec1) * np.linalg.norm(vec2) > 0 else 0
                    similarity_matrix[i, j] = similarity
        
        # 繪製相似度熱力圖
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            xticklabels=tracks,
            yticklabels=tracks
        )
        plt.title('議程軌主題相似度')
        plt.tight_layout()
        plt.savefig('visualizations/static/track_similarity.png', dpi=300)
        plt.close()
        
        # 使用階層式聚類對議程軌進行分組
        from scipy.cluster.hierarchy import linkage, dendrogram
        
        # 計算連接矩陣
        Z = linkage(similarity_matrix, 'ward')
        
        # 繪製樹狀圖
        plt.figure(figsize=(16, 10))
        dendrogram(
            Z,
            labels=tracks,
            orientation='right',
            leaf_font_size=10
        )
        plt.title('議程軌階層式聚類')
        plt.xlabel('距離')
        plt.tight_layout()
        plt.savefig('visualizations/static/track_clustering.png', dpi=300)
        plt.close()
    
    # 3.3 議程軌的時間演變分析
    if 'year' in sessions_df.columns:
        # 計算每年的議程軌數量
        yearly_track_counts = {}
        
        for year in sorted(sessions_df['year'].unique()):
            year_tracks = sessions_df[sessions_df['year'] == year]['track'].dropna().unique()
            yearly_track_counts[year] = len(year_tracks)
        
        # 分析議程軌的持續性
        track_continuity = {}
        
        for track in tracks:
            # 獲取該議程軌出現的年份
            track_years = sorted(sessions_df[sessions_df['track'] == track]['year'].unique())
            
            track_continuity[track] = {
                'years': track_years,
                'duration': len(track_years),
                'is_continuous': len(track_years) == (max(track_years) - min(track_years) + 1)
            }
        
        # 儲存議程軌時間演變資料
        with open('data/processed/track_evolution.json', 'w', encoding='utf-8') as f:
            json.dump({
                'yearly_track_counts': yearly_track_counts,
                'track_continuity': track_continuity
            }, f, ensure_ascii=False, indent=2)
        
        # 繪製議程軌數量變化圖
        plt.figure(figsize=(10, 6))
        years = sorted(yearly_track_counts.keys())
        counts = [yearly_track_counts[year] for year in years]
        
        plt.bar(years, counts, color='skyblue')
        
        # 添加數據標籤
        for i, v in enumerate(counts):
            plt.text(years[i], v + 0.5, str(v), ha='center')
        
        plt.title('COSCUP 議程軌數量變化 (2020-2024)')
        plt.xlabel('年份')
        plt.ylabel('議程軌數量')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('visualizations/static/track_count_trend.png', dpi=300)
        plt.close()
    
    return {
        'track_profiles': track_profiles,
        'track_similarity_matrix': similarity_matrix if 'similarity_matrix' in locals() else None,
        'track_continuity': track_continuity if 'track_continuity' in locals() else None
    }

# 主函數
def run_semantic_analysis(sessions_file='data/processed/sessions.csv', 
                         speakers_file='data/processed/speakers.csv'):
    """運行所有語意分析"""
    
    print(f"載入資料: {sessions_file} 和 {speakers_file}")
    
    # 載入資料
    sessions_df = pd.read_csv(sessions_file)
    try:
        speakers_df = pd.read_csv(speakers_file)
    except:
        print(f"警告: 無法載入講者資料 {speakers_file}")
        speakers_df = pd.DataFrame()
    
    # 確保輸出目錄存在
    os.makedirs('visualizations/static', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # 執行語意分析與關鍵詞提取
    print("\n=== 開始語意分析與關鍵詞提取 ===")
    semantic_results = semantic_analysis(sessions_df, speakers_df)
    
    # 執行講者與議程的語意關係分析
    print("\n=== 開始講者與議程的語意關係分析 ===")
    speaker_session_results = speaker_session_semantic_analysis(sessions_df, speakers_df)
    
    # 執行議程軌語意分析
    print("\n=== 開始議程軌語意分析 ===")
    track_results = track_semantic_analysis(sessions_df)
    
    print("\n所有語意分析完成！")
    return {
        'semantic_results': semantic_results,
        'speaker_session_results': speaker_session_results,
        'track_results': track_results
    }

if __name__ == "__main__":
    run_semantic_analysis()
