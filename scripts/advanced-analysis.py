import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import networkx as nx
import community as community_louvain
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import gensim
from gensim.models import Word2Vec, LdaModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import jieba
import jieba.analyse
import json
import re
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.colors as colors
from wordcloud import WordCloud

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 進階語義分析與主題建模
def advanced_topic_modeling(sessions_df):
    """使用多種主題建模方法進行比較分析"""
    print("執行進階主題建模...")
    
    # 確保有分詞後的文本
    if 'tokens' not in sessions_df.columns:
        # 載入停用詞
        stopwords = set(['的', '了', '和', '是', '就', '都', '而', '及', '與', '著',
                        '或', '一個', '我們', '你們', '他們', '可以', '這個', '那個'])
        
        # 合併標題和描述
        sessions_df['text'] = sessions_df['title'] + ' ' + sessions_df['description'].fillna('')
        
        # 分詞
        sessions_df['tokens'] = sessions_df['text'].apply(
            lambda x: [w for w in jieba.cut(x) if w not in stopwords and len(w) > 1]
        )
    
    # 準備文本資料
    docs = [' '.join(tokens) for tokens in sessions_df['tokens']]
    
    # 1.1 使用TF-IDF + NMF進行主題建模
    print("使用TF-IDF + NMF進行主題建模...")
    n_topics = 8
    n_top_words = 10
    
    # TF-IDF矢量化
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
    tfidf = tfidf_vectorizer.fit_transform(docs)
    
    # 使用NMF (Non-negative Matrix Factorization)
    nmf_model = NMF(n_components=n_topics, random_state=42)
    nmf_topic_doc_matrix = nmf_model.fit_transform(tfidf)
    
    # 獲取特徵名稱
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # 將NMF主題-詞項矩陣轉換為每個主題的關鍵詞及其權重
    nmf_topics = {}
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [(tfidf_feature_names[i], topic[i]) for i in top_features_ind]
        nmf_topics[f"topic_{topic_idx}"] = dict(top_features)
    
    # 1.2 使用CountVectorizer + LDA進行主題建模
    print("使用CountVectorizer + LDA進行主題建模...")
    
    # 詞頻向量化
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
    count_data = count_vectorizer.fit_transform(docs)
    
    # 使用LDA (Latent Dirichlet Allocation)
    lda_model = LatentDirichletAllocation(
        n_components=n_topics, 
        max_iter=10, 
        learning_method='online',
        random_state=42
    )
    lda_topic_doc_matrix = lda_model.fit_transform(count_data)
    
    # 獲取特徵名稱
    count_feature_names = count_vectorizer.get_feature_names_out()
    
    # 將LDA主題-詞項矩陣轉換為每個主題的關鍵詞及其權重
    lda_topics = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [(count_feature_names[i], topic[i]) for i in top_features_ind]
        lda_topics[f"topic_{topic_idx}"] = dict(top_features)
    
    # 1.3 使用Gensim進行LDA主題建模
    print("使用Gensim進行LDA主題建模...")
    
    # 準備Gensim格式的資料
    dictionary = gensim.corpora.Dictionary(sessions_df['tokens'])
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(tokens) for tokens in sessions_df['tokens']]
    
    # 訓練LDA模型
    gensim_lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=10,
        alpha='auto',
        random_state=42
    )
    
    # 獲取主題詞及其權重
    gensim_lda_topics = {}
    for topic_id in range(n_topics):
        top_words = gensim_lda_model.show_topic(topic_id, topn=n_top_words)
        gensim_lda_topics[f"topic_{topic_id}"] = dict(top_words)
    
    # 獲取每個文檔的主題分布
    doc_topics = []
    for i, doc_bow in enumerate(corpus):
        topics = gensim_lda_model.get_document_topics(doc_bow)
        # 取主要主題
        main_topic = max(topics, key=lambda x: x[1]) if topics else (0, 0)
        
        doc_topics.append({
            'session_id': sessions_df.iloc[i]['id'],
            'year': sessions_df.iloc[i]['year'],
            'title': sessions_df.iloc[i]['title'],
            'main_topic': main_topic[0],
            'topic_prob': float(main_topic[1]),
            'topic_dist': {str(tid): float(tprob) for tid, tprob in topics}
        })
    
    doc_topics_df = pd.DataFrame(doc_topics)
    
    # 1.4 主題模型的可視化
    print("主題模型的可視化...")
    
    # 主題關鍵詞雲圖
    for model_name, topics in [("NMF", nmf_topics), ("LDA", lda_topics), ("Gensim_LDA", gensim_lda_topics)]:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for topic_idx, topic_dict in enumerate(topics.values()):
            if topic_idx < 8:
                ax = axes[topic_idx]
                wordcloud = WordCloud(
                    width=400, height=400,
                    background_color='white',
                    max_words=50,
                    font_path='simhei.ttf'  # 需確保有合適的中文字體
                )
                wordcloud.generate_from_frequencies(topic_dict)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'Topic {topic_idx+1}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'visualizations/static/{model_name}_topic_wordclouds.png', dpi=300)
        plt.close()
    
    # 主題趨勢圖
    yearly_topic_counts = doc_topics_df.groupby(['year', 'main_topic']).size().unstack().fillna(0)
    yearly_topic_perc = yearly_topic_counts.div(yearly_topic_counts.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(12, 8))
    for topic in yearly_topic_perc.columns:
        plt.plot(yearly_topic_perc.index, yearly_topic_perc[topic], marker='o', label=f'Topic {topic}')
    
    plt.title('COSCUP 議程主題趨勢變化 (2020-2024)')
    plt.xlabel('年份')
    plt.ylabel('比例 (%)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/static/topic_trends.png', dpi=300)
    plt.close()
    
    return {
        'nmf_topics': nmf_topics,
        'lda_topics': lda_topics,
        'gensim_lda_topics': gensim_lda_topics,
        'doc_topics_df': doc_topics_df,
        'yearly_topic_perc': yearly_topic_perc
    }

# 2. 講者嵌入與聚類
def speaker_embedding_and_clustering(sessions_df, speakers_df):
    """使用Doc2Vec為講者創建嵌入向量，並進行聚類分析"""
    print("執行講者嵌入與聚類分析...")
    
    # 建立講者到議程的映射
    speaker_to_sessions = defaultdict(list)
    for _, row in sessions_df.iterrows():
        # 假設speakers欄位是講者ID的列表
        if isinstance(row['speakers'], list):
            for speaker_id in row['speakers']:
                speaker_to_sessions[speaker_id].append(row)
        elif isinstance(row['speakers'], str):
            # 嘗試解析JSON字符串
            try:
                speaker_ids = json.loads(row['speakers'])
                if isinstance(speaker_ids, list):
                    for speaker_id in speaker_ids:
                        speaker_to_sessions[speaker_id].append(row)
            except:
                # 可能是逗號分隔的字符串
                for speaker_id in row['speakers'].split(','):
                    speaker_id = speaker_id.strip()
                    if speaker_id:
                        speaker_to_sessions[speaker_id].append(row)
    
    # 合併每位講者的所有議程文本
    speaker_texts = {}
    for speaker_id, sessions in speaker_to_sessions.items():
        combined_text = ' '.join([
            f"{s['title']} {s.get('description', '')}" 
            for s in sessions if isinstance(s, dict) or isinstance(s, pd.Series)
        ])
        if combined_text.strip():
            speaker_texts[speaker_id] = combined_text
    
    # 分詞處理
    speaker_tokens = {}
    for speaker_id, text in speaker_texts.items():
        # 使用jieba分詞
        tokens = [w for w in jieba.cut(text) if len(w) > 1]
        if tokens:
            speaker_tokens[speaker_id] = tokens
    
    # 準備Doc2Vec訓練資料
    tagged_data = [TaggedDocument(words=tokens, tags=[speaker_id]) 
                  for speaker_id, tokens in speaker_tokens.items()]
    
    # 訓練Doc2Vec模型
    model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=40)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    
    # 獲取每位講者的向量表示
    speaker_vectors = {speaker_id: model.dv[speaker_id] for speaker_id in speaker_tokens.keys()}
    
    # 準備聚類資料
    speaker_ids = list(speaker_vectors.keys())
    X = np.array([speaker_vectors[sid] for sid in speaker_ids])
    
    # 使用K-means聚類
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    speaker_clusters = kmeans.fit_predict(X)
    
    # 使用t-SNE降維
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # 建立講者嵌入和聚類資料框
    speaker_embedding_df = pd.DataFrame({
        'speaker_id': speaker_ids,
        'cluster': speaker_clusters,
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1]
    })
    
    # 合併講者資訊
    if not speakers_df.empty:
        speaker_info = speakers_df.set_index('id')[['name', 'company']].to_dict('index')
        speaker_embedding_df['name'] = speaker_embedding_df['speaker_id'].map(
            lambda sid: speaker_info.get(sid, {}).get('name', f'Speaker {sid}')
        )
        speaker_embedding_df['company'] = speaker_embedding_df['speaker_id'].map(
            lambda sid: speaker_info.get(sid, {}).get('company', '')
        )
    
    # 可視化講者嵌入和聚類
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        speaker_embedding_df['x'], 
        speaker_embedding_df['y'],
        c=speaker_embedding_df['cluster'], 
        cmap='tab10',
        alpha=0.7,
        s=50
    )
    
    # 標記重要講者 (假設session數量>2的為重要講者)
    important_speakers = []
    for speaker_id in speaker_ids:
        if len(speaker_to_sessions.get(speaker_id, [])) > 2:
            important_speakers.append(speaker_id)
    
    for speaker_id in important_speakers:
        if speaker_id in speaker_embedding_df['speaker_id'].values:
            row = speaker_embedding_df[speaker_embedding_df['speaker_id'] == speaker_id].iloc[0]
            plt.annotate(
                row.get('name', speaker_id),
                (row['x'], row['y']),
                fontsize=9,
                alpha=0.8,
                xytext=(5, 5),
                textcoords='offset points'
            )
    
    plt.colorbar(scatter, label='Cluster')
    plt.title('講者嵌入向量空間 (t-SNE降維)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig('visualizations/static/speaker_embedding.png', dpi=300)
    plt.close()
    
    # 分析每個聚類的特點
    cluster_analysis = {}
    for cluster in range(n_clusters):
        cluster_speakers = speaker_embedding_df[speaker_embedding_df['cluster'] == cluster]
        
        # 計算該聚類的講者公司分布
        if 'company' in cluster_speakers.columns:
            company_counts = cluster_speakers['company'].value_counts().head(5).to_dict()
        else:
            company_counts = {}
        
        # 提取該聚類講者的議程主題
        cluster_speaker_ids = cluster_speakers['speaker_id'].tolist()
        cluster_sessions = []
        for speaker_id in cluster_speaker_ids:
            cluster_sessions.extend(speaker_to_sessions.get(speaker_id, []))
        
        # 提取主題關鍵字 (對議程標題使用TF-IDF)
        if cluster_sessions:
            # 將session轉換為DataFrame以便操作
            if isinstance(cluster_sessions[0], dict):
                cluster_sessions_df = pd.DataFrame(cluster_sessions)
            else:  # 假設已經是pd.Series
                cluster_sessions_df = pd.DataFrame(cluster_sessions.tolist())
            
            if 'title' in cluster_sessions_df.columns:
                titles = cluster_sessions_df['title'].dropna().tolist()
                
                # 提取關鍵詞
                if titles:
                    all_titles = ' '.join(titles)
                    keywords = jieba.analyse.extract_tags(all_titles, topK=10, withWeight=True)
                    keywords_dict = {word: score for word, score in keywords}
                else:
                    keywords_dict = {}
            else:
                keywords_dict = {}
        else:
            keywords_dict = {}
        
        cluster_analysis[cluster] = {
            'size': len(cluster_speakers),
            'companies': company_counts,
            'keywords': keywords_dict
        }
    
    # 將聚類分析保存為JSON
    with open('data/processed/speaker_clusters.json', 'w', encoding='utf-8') as f:
        json.dump(cluster_analysis, f, ensure_ascii=False, indent=2)
    
    return {
        'speaker_embedding_df': speaker_embedding_df,
        'cluster_analysis': cluster_analysis
    }

# 3. 時間序列分析
def time_series_analysis(sessions_df):
    """分析議程數量、主題分布等隨時間的變化"""
    print("執行時間序列分析...")
    
    # 確保有正確的時間列
    if 'start' in sessions_df.columns:
        # 轉換時間格式
        sessions_df['start_datetime'] = pd.to_datetime(sessions_df['start'])
        sessions_df['year'] = sessions_df['start_datetime'].dt.year
        sessions_df['month'] = sessions_df['start_datetime'].dt.month
        sessions_df['day'] = sessions_df['start_datetime'].dt.day
        sessions_df['hour'] = sessions_df['start_datetime'].dt.hour
        sessions_df['minute'] = sessions_df['start_datetime'].dt.minute
        sessions_df['dayofweek'] = sessions_df['start_datetime'].dt.dayofweek
    
    # 3.1 按小時分析議程分布
    if 'hour' in sessions_df.columns:
        hourly_counts = sessions_df.groupby(['year', 'hour']).size().unstack(fill_value=0)
        
        # 繪製熱力圖
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(hourly_counts, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('議程時間分布 (按小時)')
        plt.xlabel('小時')
        plt.ylabel('年份')
        plt.tight_layout()
        plt.savefig('visualizations/static/hourly_distribution.png', dpi=300)
        plt.close()
    
    # 3.2 按星期幾和小時分析議程密度
    if 'dayofweek' in sessions_df.columns and 'hour' in sessions_df.columns:
        # 合併所有年份的資料
        day_hour_counts = sessions_df.groupby(['dayofweek', 'hour']).size().unstack(fill_value=0)
        
        # 繪製熱力圖
        plt.figure(figsize=(12, 6))
        day_names = ['週一', '週二', '週三', '週四', '週五', '週六', '週日']
        ax = sns.heatmap(day_hour_counts, annot=True, fmt='d', cmap='YlGnBu',
                         yticklabels=[day_names[i] for i in day_hour_counts.index])
        plt.title('議程時間分布 (按星期和小時)')
        plt.xlabel('小時')
        plt.ylabel('星期')
        plt.tight_layout()
        plt.savefig('visualizations/static/day_hour_distribution.png', dpi=300)
        plt.close()
    
    # 3.3 分析會議規模的年度變化
    yearly_counts = sessions_df.groupby('year').size()
    
    plt.figure(figsize=(10, 6))
    yearly_counts.plot(kind='bar', color='skyblue')
    for i, v in enumerate(yearly_counts):
        plt.text(i, v + 1, str(v), ha='center')
    plt.title('COSCUP 議程數量年度變化')
    plt.xlabel('年份')
    plt.ylabel('議程數量')
    plt.tight_layout()
    plt.savefig('visualizations/static/yearly_session_counts.png', dpi=300)
    plt.close()
    
    # 3.4 如果有議程類型欄位，分析各類型議程的年度變化
    if 'type' in sessions_df.columns:
        type_yearly_counts = sessions_df.groupby(['year', 'type']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 8))
        type_yearly_counts.plot(kind='bar', stacked=True)
        plt.title('各類型議程數量年度變化')
        plt.xlabel('年份')
        plt.ylabel('議程數量')
        plt.legend(title='議程類型')
        plt.tight_layout()
        plt.savefig('visualizations/static/type_yearly_counts.png', dpi=300)
        plt.close()
    
    # 3.5 議程時長分析
    if 'start' in sessions_df.columns and 'end' in sessions_df.columns:
        # 計算議程時長（分鐘）
        sessions_df['end_datetime'] = pd.to_datetime(sessions_df['end'])
        sessions_df['duration_minutes'] = (sessions_df['end_datetime'] - sessions_df['start_datetime']).dt.total_seconds() / 60
        
        # 按年份繪製時長分布箱形圖
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='year', y='duration_minutes', data=sessions_df)
        plt.title('議程時長分布 (按年份)')
        plt.xlabel('年份')
        plt.ylabel('時長 (分鐘)')
        plt.tight_layout()
        plt.savefig('visualizations/static/duration_boxplot.png', dpi=300)
        plt.close()
        
        # 時長分布直方圖
        plt.figure(figsize=(10, 6))
        sns.histplot(sessions_df['duration_minutes'], bins=20, kde=True)
        plt.title('議程時長分布')
        plt.xlabel('時長 (分鐘)')
        plt.ylabel('議程數量')
        plt.tight_layout()
        plt.savefig('visualizations/static/duration_histogram.png', dpi=300)
        plt.close()
    
    return {
        'hourly_counts': hourly_counts if 'hourly_counts' in locals() else None,
        'day_hour_counts': day_hour_counts if 'day_hour_counts' in locals() else None,
        'yearly_counts': yearly_counts,
        'type_yearly_counts': type_yearly_counts if 'type_yearly_counts' in locals() else None
    }

# 4. 複雜網絡分析
def complex_network_analysis(sessions_df, speakers_df):
    """進行更深入的講者和議程軌之間的網絡分析"""
    print("執行複雜網絡分析...")
    
    # 4.1 創建講者合作網絡
    G_speakers = nx.Graph()
    
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
    
    # 添加節點
    for speaker_id, session_ids in speaker_to_sessions.items():
        # 查找講者資訊
        speaker_info = {}
        if not speakers_df.empty and 'id' in speakers_df.columns:
            speaker_row = speakers_df[speakers_df['id'] == speaker_id]
            if not speaker_row.empty:
                for col in speaker_row.columns:
                    if col != 'id':
                        speaker_info[col] = speaker_row.iloc[0][col]
        
        G_speakers.add_node(speaker_id, 
                          session_count=len(session_ids),
                          **speaker_info)
    
    # 添加邊 - 如果兩位講者共同參與過議程則建立連接
    for _, row in sessions_df.iterrows():
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
        
        if len(speakers) > 1:
            for i in range(len(speakers)):
                for j in range(i+1, len(speakers)):
                    speaker1 = speakers[i]
                    speaker2 = speakers[j]
                    if G_speakers.has_edge(speaker1, speaker2):
                        G_speakers[speaker1][speaker2]['weight'] += 1
                    else:
                        G_speakers.add_edge(speaker1, speaker2, weight=1)
    
    # 4.2 計算網絡指標
    print("計算網絡指標...")
    
    # 計算度中心性
    degree_centrality = nx.degree_centrality(G_speakers)
    # 計算介數中心性
    betweenness_centrality = nx.betweenness_centrality(G_speakers)
    # 計算接近中心性
    closeness_centrality = nx.closeness_centrality(G_speakers)
    # 計算特徵向量中心性
    eigenvector_centrality = nx.eigenvector_centrality(G_speakers, max_iter=1000)
    
    # 更新節點屬性
    for node in G_speakers.nodes():
        G_speakers.nodes[node]['degree_centrality'] = degree_centrality[node]
        G_speakers.nodes[node]['betweenness_centrality'] = betweenness_centrality[node]
        G_speakers.nodes[node]['closeness_centrality'] = closeness_centrality[node]
        G_speakers.nodes[node]['eigenvector_centrality'] = eigenvector_centrality[node]
    
    # 4.3 社群檢測
    print("進行社群檢測...")
    
    # 使用Louvain算法進行社群檢測
    communities = community_louvain.best_partition(G_speakers)
    
    # 更新節點屬性
    for node, community in communities.items():
        G_speakers.nodes[node]['community'] = community
    
    # 4.4 可視化網絡
    print("可視化網絡...")
    
    # 設置繪圖參數
    plt.figure(figsize=(16, 16))
    
    # 獲取唯一的社群ID
    unique_communities = set(communities.values())
    color_map = plt.cm.tab20(np.linspace(0, 1, len(unique_communities)))
    community_colors = {comm: color_map[i] for i, comm in enumerate(unique_communities)}
    
    # 節點位置
    pos = nx.spring_layout(G_speakers, k=0.3, iterations=50, seed=42)
    
    # 繪製邊
    for u, v, weight in G_speakers.edges(data='weight'):
        nx.draw_networkx_edges(
            G_speakers, pos, 
            edgelist=[(u, v)], 
            width=np.sqrt(weight),
            alpha=0.5,
            edge_color='lightgray'
        )
    
    # 繪製節點
    for community in unique_communities:
        # 篩選該社群的節點
        nodelist = [node for node, comm in communities.items() if comm == community]
        
        # 根據節點屬性調整節點大小
        node_sizes = [100 + G_speakers.nodes[node].get('session_count', 1) * 30 for node in nodelist]
        
        # 繪製節點
        nx.draw_networkx_nodes(
            G_speakers, pos,
            nodelist=nodelist,
            node_size=node_sizes,
            node_color=[community_colors[community]] * len(nodelist),
            alpha=0.7,
            label=f'社群 {community}'
        )
    
    # 為重要節點添加標籤
    # 選擇度中心性排名前20的節點
    top_nodes = sorted(
        G_speakers.nodes(), 
        key=lambda x: G_speakers.nodes[x].get('degree_centrality', 0),
        reverse=True
    )[:20]
    
    labels = {node: G_speakers.nodes[node].get('name', node) for node in top_nodes}
    nx.draw_networkx_labels(G_speakers, pos, labels=labels, font_size=10, font_family='SimHei')
    
    plt.title('COSCUP 講者合作網絡圖 (按社群著色)', fontsize=16)
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/static/speaker_network_communities.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4.5 建立講者-議程軌二分網絡
    if 'track' in sessions_df.columns:
        print("建立講者-議程軌二分網絡...")
        
        B = nx.Graph()
        
        # 添加講者節點
        for speaker_id in speaker_to_sessions:
            B.add_node(speaker_id, bipartite=0, type='speaker')
        
        # 添加議程軌節點
        tracks = sessions_df['track'].unique()
        for track in tracks:
            if pd.notna(track) and track:
                B.add_node(str(track), bipartite=1, type='track')
        
        # 添加邊 - 如果講者在該議程軌有議程
        for speaker_id, session_ids in speaker_to_sessions.items():
            # 獲取該講者所有議程
            speaker_sessions = sessions_df[sessions_df['id'].isin(session_ids)]
            
            # 統計每個議程軌的議程數量
            track_counts = speaker_sessions['track'].value_counts()
            
            # 添加邊
            for track, count in track_counts.items():
                if pd.notna(track) and track:
                    B.add_edge(speaker_id, str(track), weight=count)
        
        # 可視化二分網絡
        plt.figure(figsize=(16, 12))
        
        # 獲取節點集合
        speakers = {n for n, d in B.nodes(data=True) if d['bipartite'] == 0}
        tracks = {n for n, d in B.nodes(data=True) if d['bipartite'] == 1}
        
        # 使用bipartite_layout設置位置
        pos = nx.bipartite_layout(B, speakers)
        
        # 繪製邊
        edge_weights = [B[u][v]['weight'] * 0.5 for u, v in B.edges()]
        nx.draw_networkx_edges(B, pos, width=edge_weights, alpha=0.3, edge_color='gray')
        
        # 繪製講者節點
        nx.draw_networkx_nodes(
            B, pos,
            nodelist=speakers,
            node_color='skyblue',
            node_size=50,
            alpha=0.8,
            label='講者'
        )
        
        # 繪製議程軌節點 - 根據連接數調整大小
        track_sizes = [100 + B.degree(track) * 5 for track in tracks]
        nx.draw_networkx_nodes(
            B, pos,
            nodelist=tracks,
            node_color='orange',
            node_size=track_sizes,
            alpha=0.8,
            label='議程軌'
        )
        
        # 為議程軌添加標籤
        track_labels = {track: track for track in tracks}
        nx.draw_networkx_labels(B, pos, labels=track_labels, font_size=8, font_family='SimHei')
        
        plt.title('講者-議程軌 二分網絡圖', fontsize=16)
        plt.axis('off')
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/static/speaker_track_bipartite.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4.6 儲存網絡分析結果
    print("儲存網絡分析結果...")
    
    # 提取頂級講者
    top_speakers = []
    for node in G_speakers.nodes():
        if G_speakers.nodes[node].get('degree_centrality', 0) > 0.1:  # 設置一個閾值
            speaker_info = dict(G_speakers.nodes[node])
            speaker_info['id'] = node
            top_speakers.append(speaker_info)
    
    # 將結果保存為CSV
    top_speakers_df = pd.DataFrame(top_speakers)
    if not top_speakers_df.empty:
        top_speakers_df.to_csv('data/processed/top_speakers_network.csv', index=False, encoding='utf-8')
    
    # 計算每個社群的統計資料
    community_stats = defaultdict(lambda: {'count': 0, 'companies': Counter(), 'avg_degree': 0})
    
    for node, comm in communities.items():
        community_stats[comm]['count'] += 1
        if 'company' in G_speakers.nodes[node] and G_speakers.nodes[node]['company']:
            community_stats[comm]['companies'][G_speakers.nodes[node]['company']] += 1
        community_stats[comm]['avg_degree'] += G_speakers.degree(node)
    
    # 計算平均度
    for comm in community_stats:
        if community_stats[comm]['count'] > 0:
            community_stats[comm]['avg_degree'] /= community_stats[comm]['count']
        
        # 轉換公司計數器為字典
        community_stats[comm]['companies'] = dict(community_stats[comm]['companies'].most_common(5))
    
    # 保存社群統計資料
    with open('data/processed/community_stats.json', 'w', encoding='utf-8') as f:
        json.dump(community_stats, f, ensure_ascii=False, indent=2)
    
    return {
        'speaker_graph': G_speakers,
        'communities': communities,
        'community_stats': community_stats,
        'top_speakers_df': top_speakers_df,
        'bipartite_graph': B if 'B' in locals() else None
    }

# 主函數
def run_advanced_analysis(sessions_file='data/processed/sessions.csv', 
                         speakers_file='data/processed/speakers.csv'):
    """運行所有進階分析"""
    
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
    
    # 執行進階主題建模
    print("\n=== 開始進階主題建模 ===")
    topic_results = advanced_topic_modeling(sessions_df)
    
    # 執行講者嵌入與聚類
    print("\n=== 開始講者嵌入與聚類 ===")
    embedding_results = speaker_embedding_and_clustering(sessions_df, speakers_df)
    
    # 執行時間序列分析
    print("\n=== 開始時間序列分析 ===")
    time_series_results = time_series_analysis(sessions_df)
    
    # 執行複雜網絡分析
    print("\n=== 開始複雜網絡分析 ===")
    network_results = complex_network_analysis(sessions_df, speakers_df)
    
    print("\n所有進階分析完成！")
    return {
        'topic_results': topic_results,
        'embedding_results': embedding_results,
        'time_series_results': time_series_results,
        'network_results': network_results
    }

if __name__ == "__main__":
    run_advanced_analysis()
