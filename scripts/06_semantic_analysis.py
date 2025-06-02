import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import json
import os
import utils

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                   'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def semantic_analysis(sessions_df, speakers_df):
    """對議程內容和講者背景進行語意分析"""
    print("進行語意分析與關鍵詞提取...")
    # 1.1 分析議程標題和描述
    if 'title' in sessions_df.columns:
        # 分詞
        sessions_df = utils.tokenize_text(sessions_df, 'title')
        # 為每個議程提取關鍵詞
        # Note: extract_keywords in utils.py returns a dict of yearly keywords
        # This part needs to be adjusted if 'keywords' column is expected
        # For now, let's assume utils.extract_keywords is adapted or
        # we use a different approach for individual session keywords.
        # Placeholder for individual session keywords:
        sessions_df['keywords'] = sessions_df['text'].apply(
            lambda x: utils.extract_keywords(
                pd.DataFrame([{'year': 2000, 'tokens': x.split()}]), # Dummy DF
                'tokens', top_n=8
            ).get(2000, {}).keys() # Extract keys for the dummy year
        )

        # 繪製整體關鍵詞雲
        all_keywords_list = []
        for keywords_set in sessions_df['keywords']:
            all_keywords_list.extend(list(keywords_set))

        keyword_counts = Counter(all_keywords_list)
        if keyword_counts:  # Ensure there are keywords to plot
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
    # 1.2 分析講者背景
    if not speakers_df.empty and 'bio' in speakers_df.columns:
        # 分詞和提取關鍵詞
        speakers_df = utils.tokenize_text(speakers_df, 'bio')
        # Placeholder for individual speaker bio keywords:
        speakers_df['bio_keywords'] = speakers_df['text'].apply(
            lambda x: utils.extract_keywords(
                 pd.DataFrame([{'year': 2000, 'tokens': x.split()}]), # Dummy DF
                'tokens', top_n=5
            ).get(2000, {}).keys() # Extract keys for the dummy year
        )
        # 繪製講者背景關鍵詞雲
        all_bio_keywords_list = []
        for keywords_set in speakers_df['bio_keywords']:
            all_bio_keywords_list.extend(list(keywords_set))

        bio_keyword_counts = Counter(all_bio_keywords_list)
        if bio_keyword_counts:  # Ensure there are keywords to plot
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
            plt.savefig('visualizations/static/speaker_bio_wordcloud.png',
                        dpi=300)
            plt.close()
    # 1.3 議程相似度分析
    similar_sessions = []  # Initialize similar_sessions
    if 'tokens' in sessions_df.columns:
        # 建立文檔-詞項矩陣
        docs = [' '.join(tokens) for tokens in sessions_df['tokens']]
        if docs:  # Ensure there are documents to process
            vectorizer = TfidfVectorizer(max_features=500)
            X = vectorizer.fit_transform(docs)
            # 計算議程之間的相似度
            similarity_matrix = cosine_similarity(X)
            # 找出每個議程最相似的其他議程
            for i in range(len(sessions_df)):
                sim_scores = list(enumerate(similarity_matrix[i]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1],
                                    reverse=True)
                # 排除自己
                sim_scores = sim_scores[1:6]  # 取前5個最相似的
                similar_indices = [
                    idx for idx, score in sim_scores if score > 0.3
                ]  # 設置一個閾值
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
            with open('data/processed/similar_sessions.json', 'w',
                      encoding='utf-8') as f:
                json.dump(similar_sessions, f, ensure_ascii=False, indent=2)
    # 1.4 技術關鍵詞趨勢分析
    tech_trends_df = None  # Initialize tech_trends_df
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
        for year_val in years:  # Renamed to avoid conflict
            year_docs = sessions_df[sessions_df['year'] == year_val]
            year_tokens = []
            for tokens_list in year_docs['tokens']:  # Renamed
                year_tokens.extend(tokens_list)
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
        tech_trends_df.to_csv('data/processed/tech_trends.csv',
                              encoding='utf-8')
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
        'session_keywords': (
            sessions_df['keywords'].tolist()
            if 'keywords' in sessions_df.columns else []
        ),
        'tech_trends_df': tech_trends_df,
        'similar_sessions': similar_sessions
    }


def speaker_session_semantic_analysis(sessions_df, speakers_df):
    """分析講者與議程之間的語意關係"""
    print("分析講者與議程的語意關係...")
    # 建立講者到議程的映射
    speaker_to_sessions = defaultdict(list)
    for _, row in sessions_df.iterrows():
        # 假設speakers欄位是講者ID的列表或字符串
        speakers_data = []
        if isinstance(row['speakers'], list):
            speakers_data = row['speakers']
        elif isinstance(row['speakers'], str):
            try:
                speakers_data = json.loads(row['speakers'])
                if not isinstance(speakers_data, list):
                    speakers_data = row['speakers'].split(',')
            except json.JSONDecodeError:
                speakers_data = row['speakers'].split(',')
        speakers_data = [s.strip() for s in speakers_data if s.strip()]
        for speaker_id in speakers_data:
            speaker_to_sessions[speaker_id].append(row['id'])
    # 2.1 講者專長領域分析
    speaker_expertise = {}
    if 'keywords' in sessions_df.columns:
        for speaker_id, session_ids in speaker_to_sessions.items():
            # 取得該講者的所有議程關鍵詞
            speaker_sessions_df = sessions_df[sessions_df['id'].isin(session_ids)]
            all_keywords_list = []
            for kw_set in speaker_sessions_df['keywords']: # Renamed
                all_keywords_list.extend(list(kw_set))
            # 計算關鍵詞頻率
            keyword_counter = Counter(all_keywords_list)
            # 取前10個關鍵詞作為專長領域
            top_keywords = keyword_counter.most_common(10)
            speaker_expertise[speaker_id] = {
                'top_keywords': dict(top_keywords),
                'session_count': len(session_ids),
                'years': sorted(speaker_sessions_df['year'].unique().tolist())
            }
        # 合併講者資訊
        if not speakers_df.empty:
            for speaker_id_val in speaker_expertise: # Renamed
                speaker_info = speakers_df[speakers_df['id'] == speaker_id_val]
                if not speaker_info.empty:
                    if 'name' in speaker_info.columns:
                        speaker_expertise[speaker_id_val]['name'] = \
                            speaker_info.iloc[0]['name']
                    if 'company' in speaker_info.columns:
                        speaker_expertise[speaker_id_val]['company'] = \
                            speaker_info.iloc[0]['company']
        # 儲存講者專長資料
        with open('data/processed/speaker_expertise.json', 'w',
                  encoding='utf-8') as f:
            json.dump(speaker_expertise, f, ensure_ascii=False, indent=2)
    # 2.2 共同主題講者群組分析
    keyword_speakers = None # Initialize
    if 'keywords' in sessions_df.columns and speaker_expertise: # Check speaker_expertise
        # 根據關鍵詞將講者分組
        keyword_to_speakers_map = defaultdict(set) # Renamed
        for speaker_id_val, info_val in speaker_expertise.items(): # Renamed
            for keyword_item in info_val['top_keywords']: # Renamed
                keyword_to_speakers_map[keyword_item].add(speaker_id_val)
        # 取前20個最多講者的關鍵詞
        popular_keywords_list = sorted( # Renamed
            keyword_to_speakers_map.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:20]
        # 繪製每個關鍵詞的講者數量
        if popular_keywords_list: # Check if list is not empty
            plt.figure(figsize=(14, 8))
            keywords_plot = [kw for kw, _ in popular_keywords_list] # Renamed
            speaker_counts_plot = [len(spkrs) for _, spkrs in popular_keywords_list] # Renamed
            plt.bar(keywords_plot, speaker_counts_plot, color='skyblue')
            plt.xticks(rotation=45, ha='right')
            plt.title('熱門技術關鍵詞講者數量')
            plt.xlabel('關鍵詞')
            plt.ylabel('講者數量')
            plt.tight_layout()
            plt.savefig('visualizations/static/keyword_speaker_counts.png',
                        dpi=300)
            plt.close()
        # 儲存關鍵詞-講者映射
        keyword_speakers = {
            kw_item: list(spkrs_set) # Renamed
            for kw_item, spkrs_set in keyword_to_speakers_map.items()
            if len(spkrs_set) >= 3  # 至少3位講者
        }
        with open('data/processed/keyword_speakers.json', 'w',
                  encoding='utf-8') as f:
            json.dump(keyword_speakers, f, ensure_ascii=False, indent=2)
    # 2.3 講者主題演變分析
    recurring_speakers_list = [] # Renamed
    speaker_evolution = None # Initialize
    for speaker_id_val, info_val in speaker_expertise.items(): # Renamed
        # 找出參與多年的講者
        if len(info_val.get('years', [])) > 1:
            recurring_speakers_list.append(speaker_id_val)
    if recurring_speakers_list and 'keywords' in sessions_df.columns:
        # 分析每位重複出席講者的主題變化
        speaker_evolution_map = {} # Renamed
        for speaker_id_val in recurring_speakers_list: # Renamed
            speaker_sessions_map = {} # Renamed
            # 獲取該講者每年的議程
            for session_id_val in speaker_to_sessions[speaker_id_val]: # Renamed
                session_item = sessions_df[
                    sessions_df['id'] == session_id_val
                ].iloc[0] # Renamed
                year_val = session_item['year'] # Renamed
                if year_val not in speaker_sessions_map:
                    speaker_sessions_map[year_val] = []
                speaker_sessions_map[year_val].append({
                    'id': session_item['id'],
                    'title': session_item['title'],
                    'keywords': session_item['keywords']
                })
            # 計算每年的關鍵詞
            yearly_keywords_map = {} # Renamed
            for year_item, sessions_list in speaker_sessions_map.items(): # Renamed
                all_keywords_yearly = [] # Renamed
                for session_data_item in sessions_list:  # Renamed
                    all_keywords_yearly.extend(list(session_data_item['keywords']))
                keyword_counter_yearly = Counter(all_keywords_yearly) # Renamed
                yearly_keywords_map[year_item] = dict(
                    keyword_counter_yearly.most_common(5)
                )
            # 添加講者名稱
            name_val = speaker_expertise[speaker_id_val].get('name', speaker_id_val) # Renamed
            speaker_evolution_map[speaker_id_val] = {
                'name': name_val,
                'yearly_keywords': yearly_keywords_map,
                'years': sorted(yearly_keywords_map.keys())
            }
        speaker_evolution = speaker_evolution_map # Assign to outer scope variable
        # 儲存講者主題演變資料
        with open('data/processed/speaker_evolution.json', 'w',
                  encoding='utf-8') as f:
            json.dump(speaker_evolution_map, f, ensure_ascii=False, indent=2)
        # 繪製部分講者的關鍵詞變化
        # 選擇前5位參與年份最多的講者
        top_speakers_list = sorted( # Renamed
            speaker_evolution_map.items(),
            key=lambda x: len(x[1]['years']),
            reverse=True
        )[:5]
        for speaker_id_item, info_item in top_speakers_list: # Renamed
            plt.figure(figsize=(12, 6))
            years_list = info_item['years'] # Renamed
            yearly_keywords_data = info_item['yearly_keywords'] # Renamed
            # 找出所有出現的關鍵詞
            all_kw_set = set() # Renamed
            for _, kw_dict in yearly_keywords_data.items(): # Renamed
                all_kw_set.update(kw_dict.keys())
            # 選擇出現頻率較高的關鍵詞
            top_kw_list = [] # Renamed
            for kw_val in all_kw_set: # Renamed
                total = sum(
                    yearly_keywords_data.get(yr, {}).get(kw_val, 0)
                    for yr in years_list
                )
                if total >= 2:  # 至少出現2次
                    top_kw_list.append(kw_val)
            # 繪製關鍵詞變化趨勢
            for kw_plot in top_kw_list[:8]:  # 最多顯示8個關鍵詞, Renamed
                values = [
                    yearly_keywords_data.get(yr, {}).get(kw_plot, 0)
                    for yr in years_list
                ]
                plt.plot(years_list, values, marker='o', label=kw_plot)
            plt.title(
                f"講者 '{info_item['name']}' 主題演變 (2020-2024)"
            )
            plt.xlabel('年份')
            plt.ylabel('關鍵詞出現次數')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(
                f"visualizations/static/speaker_evolution_{speaker_id_item}.png",
                dpi=300
            )
            plt.close()
    return {
        'speaker_expertise': speaker_expertise,
        'keyword_speakers': keyword_speakers,
        'speaker_evolution': speaker_evolution
    }


def track_semantic_analysis(sessions_df):
    """分析議程軌的語意內容和特點"""
    print("進行議程軌語意分析...")
    if 'track' not in sessions_df.columns:
        print("警告: sessions_df 中沒有 track 欄位")
        return {}
    # 3.1 分析每個議程軌的關鍵詞特徵
    track_profiles = {}
    # 獲取所有議程軌
    tracks_unique = sessions_df['track'].dropna().unique() # Renamed
    for track_item in tracks_unique: # Renamed
        # 獲取該議程軌的所有議程
        track_sessions_df = sessions_df[sessions_df['track'] == track_item] # Renamed
        # 計算該議程軌的基本統計資料
        years_list = sorted(track_sessions_df['year'].unique()) # Renamed
        session_counts_map = track_sessions_df.groupby('year').size().to_dict() # Renamed
        # 提取關鍵詞 (如果有)
        top_keywords_map = {} # Renamed
        if 'keywords' in track_sessions_df.columns:
            all_keywords_track = [] # Renamed
            for kw_set in track_sessions_df['keywords']: # Renamed
                all_keywords_track.extend(list(kw_set))
            keyword_counter_track = Counter(all_keywords_track) # Renamed
            top_keywords_map = dict(keyword_counter_track.most_common(15))
        # 統計講者 (假設有speakers欄位)
        top_speakers_map = {} # Renamed
        if 'speakers' in track_sessions_df.columns:
            all_speakers_list_track = [] # Renamed
            for _, row_item in track_sessions_df.iterrows(): # Renamed
                speakers_data_track = [] # Renamed
                if isinstance(row_item['speakers'], list):
                    speakers_data_track = row_item['speakers']
                elif isinstance(row_item['speakers'], str):
                    try:
                        speakers_data_track = json.loads(row_item['speakers'])
                        if not isinstance(speakers_data_track, list):
                            speakers_data_track = row_item['speakers'].split(',')
                    except json.JSONDecodeError:
                        speakers_data_track = row_item['speakers'].split(',')
                all_speakers_list_track.extend(
                    [s.strip() for s in speakers_data_track if s.strip()]
                )
            speaker_counter_track = Counter(all_speakers_list_track) # Renamed
            top_speakers_map = dict(speaker_counter_track.most_common(10))
        track_profiles[track_item] = {
            'years': years_list,
            'session_counts': session_counts_map,
            'top_keywords': top_keywords_map,
            'top_speakers': top_speakers_map,
            'total_sessions': len(track_sessions_df)
        }
    # 儲存議程軌分析結果
    with open('data/processed/track_profiles.json', 'w',
              encoding='utf-8') as f:
        json.dump(track_profiles, f, ensure_ascii=False, indent=2)
    # 3.2 議程軌之間的相似度分析
    similarity_matrix = None # Initialize
    if 'keywords' in sessions_df.columns and track_profiles: # Check track_profiles
        # 為每個議程軌創建關鍵詞向量
        track_vectors = {}
        all_keywords_set_track = set() # Renamed
        # 收集所有關鍵詞
        for _, profile_item in track_profiles.items(): # Renamed
            all_keywords_set_track.update(profile_item['top_keywords'].keys())
        # 建立關鍵詞列表
        keyword_list_track = sorted(list(all_keywords_set_track))  # Renamed
        # 為每個議程軌創建向量
        for track_val, profile_data in track_profiles.items(): # Renamed
            vector = np.zeros(len(keyword_list_track))
            for i, keyword_val_track in enumerate(keyword_list_track): # Renamed
                vector[i] = profile_data['top_keywords'].get(keyword_val_track, 0)
            # 標準化
            if np.sum(vector) > 0:
                vector = vector / np.sum(vector)
            track_vectors[track_val] = vector
        # 計算相似度矩陣
        tracks_list_sim = sorted(track_vectors.keys())  # Renamed
        similarity_matrix_val = np.zeros( # Renamed
            (len(tracks_list_sim), len(tracks_list_sim))
        )
        for i, track1 in enumerate(tracks_list_sim):
            for j, track2 in enumerate(tracks_list_sim):
                if i == j:
                    similarity_matrix_val[i, j] = 1.0
                else:
                    vec1 = track_vectors[track1]
                    vec2 = track_vectors[track2]
                    # 計算餘弦相似度
                    norm_prod = np.linalg.norm(vec1) * np.linalg.norm(vec2)
                    similarity = np.dot(vec1, vec2) / norm_prod if norm_prod > 0 else 0
                    similarity_matrix_val[i, j] = similarity
        similarity_matrix = similarity_matrix_val # Assign to outer scope
        # 繪製相似度熱力圖
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            similarity_matrix_val,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            xticklabels=tracks_list_sim,
            yticklabels=tracks_list_sim
        )
        plt.title('議程軌主題相似度')
        plt.tight_layout()
        plt.savefig('visualizations/static/track_similarity.png', dpi=300)
        plt.close()
        # 使用階層式聚類對議程軌進行分組
        from scipy.cluster.hierarchy import linkage, dendrogram
        # 計算連接矩陣
        Z = linkage(similarity_matrix_val, 'ward')
        # 繪製樹狀圖
        plt.figure(figsize=(16, 10))
        dendrogram(
            Z,
            labels=tracks_list_sim,
            orientation='right',
            leaf_font_size=10
        )
        plt.title('議程軌階層式聚類')
        plt.xlabel('距離')
        plt.tight_layout()
        plt.savefig('visualizations/static/track_clustering.png', dpi=300)
        plt.close()
    # 3.3 議程軌的時間演變分析
    track_continuity = None # Initialize
    if 'year' in sessions_df.columns and tracks_unique is not None: # Check tracks_unique
        # 計算每年的議程軌數量
        yearly_track_counts = {}
        for year_val_evo in sorted(sessions_df['year'].unique()): # Renamed
            year_tracks_evo = sessions_df[
                sessions_df['year'] == year_val_evo
            ]['track'].dropna().unique() # Renamed
            yearly_track_counts[year_val_evo] = len(year_tracks_evo)
        # 分析議程軌的持續性
        track_continuity_map = {} # Renamed
        for track_item_evo in tracks_unique:  # Use original 'tracks_unique'
            # 獲取該議程軌出現的年份
            track_years_evo = sorted( # Renamed
                sessions_df[sessions_df['track'] == track_item_evo]['year'].unique()
            )
            track_continuity_map[track_item_evo] = {
                'years': track_years_evo,
                'duration': len(track_years_evo),
                'is_continuous': (
                    len(track_years_evo) ==
                    (max(track_years_evo) - min(track_years_evo) + 1)
                    if track_years_evo else False
                )  # Handle empty track_years
            }
        track_continuity = track_continuity_map # Assign to outer scope
        # 儲存議程軌時間演變資料
        with open('data/processed/track_evolution.json', 'w',
                  encoding='utf-8') as f:
            json.dump({
                'yearly_track_counts': yearly_track_counts,
                'track_continuity': track_continuity_map
            }, f, ensure_ascii=False, indent=2)
        # 繪製議程軌數量變化圖
        if yearly_track_counts: # Check if not empty
            plt.figure(figsize=(10, 6))
            years_plot_evo = sorted(yearly_track_counts.keys())  # Renamed
            counts_plot = [yearly_track_counts[yr] for yr in years_plot_evo] # Renamed
            plt.bar(years_plot_evo, counts_plot, color='skyblue')
            # 添加數據標籤
            for i, v_val in enumerate(counts_plot): # Renamed
                plt.text(years_plot_evo[i], v_val + 0.5, str(v_val), ha='center')
            plt.title('COSCUP 議程軌數量變化 (2020-2024)')
            plt.xlabel('年份')
            plt.ylabel('議程軌數量')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('visualizations/static/track_count_trend.png', dpi=300)
            plt.close()
    return {
        'track_profiles': track_profiles,
        'track_similarity_matrix': similarity_matrix,
        'track_continuity': track_continuity
    }


def run_semantic_analysis(
    sessions_file='data/processed/sessions.csv',
    speakers_file='data/processed/speakers.csv'
):
    """運行所有語意分析"""
    print(f"載入資料: {sessions_file} 和 {speakers_file}")
    # 載入資料
    sessions_df = pd.read_csv(sessions_file)
    try:
        speakers_df = pd.read_csv(speakers_file)
    except FileNotFoundError:  # More specific exception
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
    speaker_session_results = speaker_session_semantic_analysis(
        sessions_df, speakers_df
    )
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
