import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.font_manager as fm
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 載入停用詞
def load_stopwords():
    # 這裡可以擴充更多的中文停用詞
    stopwords = set(['的', '了', '和', '是', '就', '都', '而', '及', '與', '著',
                     '或', '一個', '我們', '你們', '他們', '可以', '這個', '那個'])
    return stopwords

# 對議程標題和描述進行分詞
def tokenize_text(sessions_df):
    stopwords = load_stopwords()
    
    # 確保jieba有載入相關技術詞彙
    tech_words = ['Kubernetes', 'Docker', 'DevOps', 'Python', 'JavaScript', 
                 'React', 'Vue', 'Angular', 'Node.js', 'Golang', 'Rust',
                 '開源', '資料科學', '人工智慧', '機器學習', '深度學習', '區塊鏈']
    for word in tech_words:
        jieba.add_word(word)
    
    # 使用 'zh' 欄位作為文本內容
    sessions_df['text'] = sessions_df['zh'].fillna('')
    
    # 分詞並移除停用詞
    sessions_df['tokens'] = sessions_df['text'].apply(
        lambda x: [w for w in jieba.cut(str(x)) if w not in stopwords and len(w) > 1]
    )
    
    return sessions_df

# 使用TF-IDF提取關鍵詞
def extract_keywords(sessions_df, top_n=20):
    # 按年份分組
    yearly_keywords = {}
    for year, group in sessions_df.groupby('year'):
        # 合併該年所有文本
        text = ' '.join([' '.join(tokens) for tokens in group['tokens']])
        # 提取關鍵詞
        keywords = jieba.analyse.extract_tags(text, topK=top_n, withWeight=True)
        yearly_keywords[year] = dict(keywords)
    
    return yearly_keywords

# 繪製年度關鍵詞熱力圖
def plot_keyword_heatmap(yearly_keywords):
    # 獲取所有年份和所有關鍵詞
    years = sorted(yearly_keywords.keys())
    all_keywords = set()
    for year_kw in yearly_keywords.values():
        all_keywords.update(year_kw.keys())
    
    # 建立熱力圖數據
    data = []
    for keyword in all_keywords:
        row = [yearly_keywords[year].get(keyword, 0) for year in years]
        data.append(row)
    
    # 建立DataFrame
    heatmap_df = pd.DataFrame(data, index=list(all_keywords), columns=years)
    
    # 繪製熱力圖
    plt.figure(figsize=(12, 16))
    sns.heatmap(heatmap_df, cmap='YlOrRd', linewidths=0.5)
    plt.title('COSCUP 議程關鍵詞年度熱力圖 (2020-2024)')
    plt.tight_layout()
    plt.savefig('keyword_heatmap.png', dpi=300)
    plt.close()

# 建立主題模型
def build_topic_model(sessions_df, n_topics=10):
    # 建立詞袋模型
    texts = sessions_df['tokens'].tolist()
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # 建立LDA模型
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=15,
        alpha='auto',
        random_state=42
    )
    
    # 印出主題
    for idx, topic in lda_model.print_topics(-1):
        print(f'主題 {idx+1}: {topic}')
    
    # 儲存互動式視覺化
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, 'topic_model.html')
    
    # 分析每個議程的主題分佈
    topic_distributions = []
    for i, doc_bow in enumerate(corpus):
        topic_dist = lda_model.get_document_topics(doc_bow)
        # 取主要主題
        main_topic = max(topic_dist, key=lambda x: x[1])
        topic_distributions.append({
            'session_id': sessions_df.iloc[i]['id'],
            'year': sessions_df.iloc[i]['year'],
            'zh': sessions_df.iloc[i]['zh'],  # 改用 'zh' 欄位
            'main_topic': main_topic[0],
            'topic_prob': main_topic[1],
            'all_topics': dict(topic_dist)
        })
    
    topic_df = pd.DataFrame(topic_distributions)
    return topic_df, lda_model

# 技術趨勢變化分析
def analyze_tech_trends(topic_df):
    # 分析每年各主題的比例
    yearly_topic_counts = topic_df.groupby(['year', 'main_topic']).size().unstack().fillna(0)
    yearly_topic_perc = yearly_topic_counts.div(yearly_topic_counts.sum(axis=1), axis=0) * 100
    
    # 繪製趨勢圖
    plt.figure(figsize=(12, 8))
    for topic in range(yearly_topic_perc.shape[1]):
        plt.plot(yearly_topic_perc.index, yearly_topic_perc[topic], marker='o', label=f'主題 {topic+1}')
    
    plt.title('COSCUP 議程主題趨勢變化 (2020-2024)')
    plt.xlabel('年份')
    plt.ylabel('比例 (%)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('topic_trends.png', dpi=300)
    plt.close()
    
    return yearly_topic_perc

# 主函數
def main():
    # 載入處理過的議程資料
    sessions_df = pd.read_csv('data/processed/sessions.csv')
    
    # 印出欄位名稱以供檢查
    print("資料欄位：", sessions_df.columns.tolist())
    
    # 分詞處理
    sessions_df = tokenize_text(sessions_df)
    
    # 提取年度關鍵詞
    yearly_keywords = extract_keywords(sessions_df, top_n=30)
    
    # 繪製關鍵詞熱力圖
    plot_keyword_heatmap(yearly_keywords)
    
    # 建立主題模型
    topic_df, lda_model = build_topic_model(sessions_df, n_topics=8)
    
    # 分析技術趨勢變化
    yearly_topic_trends = analyze_tech_trends(topic_df)
    
    # 儲存結果
    topic_df.to_csv('session_topics.csv', index=False, encoding='utf-8')
    yearly_topic_trends.to_csv('yearly_topic_trends.csv', encoding='utf-8')
    
    print("主題分析完成，已儲存視覺化結果")

if __name__ == "__main__":
    main()
