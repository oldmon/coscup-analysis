import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import community as community_louvain
from collections import Counter, defaultdict
import seaborn as sns

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 建立講者網絡
def build_speaker_network(sessions_df, speakers_df):
    # 建立講者到議程的映射
    speaker_to_sessions = defaultdict(list)
    for _, row in sessions_df.iterrows():
        # 注意：這裡假設speakers欄位是講者ID的列表，實際情況可能需要調整
        if isinstance(row['speakers'], list):
            for speaker_id in row['speakers']:
                speaker_to_sessions[speaker_id].append(row['id'])
        elif isinstance(row['speakers'], str):
            # 如果是JSON字符串，需要解析
            try:
                speaker_ids = json.loads(row['speakers'])
                if isinstance(speaker_ids, list):
                    for speaker_id in speaker_ids:
                        speaker_to_sessions[speaker_id].append(row['id'])
            except:
                # 如果不是有效的JSON，可能是以逗號分隔的字符串
                for speaker_id in row['speakers'].split(','):
                    speaker_id = speaker_id.strip()
                    if speaker_id:
                        speaker_to_sessions[speaker_id].append(row['id'])
    
    # 建立講者之間的合作關係
    G = nx.Graph()
    
    # 先印出 speakers_df 的欄位名稱以供檢查
    print("講者資料欄位：", speakers_df.columns.tolist())
    
    # 添加節點，使用 zh 作為名稱
    for _, speaker in speakers_df.iterrows():
        G.add_node(speaker['id'], 
                  name=speaker['zh'],  # 使用 zh 欄位作為講者名稱
                  sessions_count=len(speaker_to_sessions.get(speaker['id'], [])))
    
    # 添加邊 - 如果兩個講者共同參與過議程則建立連接
    edges = []
    for _, row in sessions_df.iterrows():
        if isinstance(row['speakers'], list) and len(row['speakers']) > 1:
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
                    if G.has_node(speaker1) and G.has_node(speaker2):
                        if G.has_edge(speaker1, speaker2):
                            G[speaker1][speaker2]['weight'] += 1
                        else:
                            G.add_edge(speaker1, speaker2, weight=1)
    
    return G, speaker_to_sessions

# 社群檢測
def detect_communities(G):
    # 使用Louvain算法進行社群檢測
    partition = community_louvain.best_partition(G)
    
    # 更新節點屬性
    nx.set_node_attributes(G, partition, 'community')
    
    # 計算每個社群的講者數量
    community_counts = Counter(partition.values())
    
    # 找出每個社群的主要講者
    main_speakers = {}
    for comm in set(partition.values()):
        comm_speakers = [(node, G.nodes[node]['name'], G.nodes[node].get('sessions_count', 0)) 
                        for node, node_comm in partition.items() if node_comm == comm]
        main_speakers[comm] = sorted(comm_speakers, key=lambda x: x[2], reverse=True)[:5]
    
    return partition, community_counts, main_speakers

# 繪製講者網絡圖
def plot_speaker_network(G, partition):
    plt.figure(figsize=(20, 20))
    
    # 根據社群著色
    communities = set(partition.values())
    cmap = plt.cm.get_cmap('tab20', len(communities))
    colors = [cmap(partition[node]) for node in G.nodes()]
    
    # 根據演講次數調整節點大小
    node_sizes = [50 + G.nodes[node].get('sessions_count', 1) * 20 for node in G.nodes()]
    
    # 根據合作次數調整邊的粗細
    edge_widths = [G[u][v].get('weight', 1) * 0.5 for u, v in G.edges()]
    
    # 使用spring layout繪製圖形
    pos = nx.spring_layout(G, k=0.2, iterations=50, seed=42)
    
    # 繪製節點、邊和標籤
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray')
    
    # 只為主要節點添加標籤
    big_nodes = [node for node in G.nodes() if G.nodes[node].get('sessions_count', 0) > 2]
    labels = {node: G.nodes[node]['name'] for node in big_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family='SimHei')
    
    plt.title('COSCUP 講者合作網絡 (2020-2024)', fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('speaker_network.png', dpi=300, bbox_inches='tight')
    plt.close()

# 分析重要講者
def analyze_key_speakers(G):
    # 計算各種中心性指標
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # 建立講者重要性DataFrame
    speakers_importance = pd.DataFrame({
        'speaker_id': list(G.nodes()),
        'name': [G.nodes[node].get('name', '') for node in G.nodes()],
        'company': [G.nodes[node].get('company', '') for node in G.nodes()],
        'sessions_count': [G.nodes[node].get('sessions_count', 0) for node in G.nodes()],
        'degree_centrality': [degree_centrality[node] for node in G.nodes()],
        'betweenness_centrality': [betweenness_centrality[node] for node in G.nodes()],
        'eigenvector_centrality': [eigenvector_centrality[node] for node in G.nodes()],
        'community': [G.nodes[node].get('community', -1) for node in G.nodes()]
    })
    
    # 計算綜合得分
    speakers_importance['importance_score'] = (
        speakers_importance['degree_centrality'] + 
        speakers_importance['betweenness_centrality'] + 
        speakers_importance['eigenvector_centrality']
    ) / 3
    
    # 排序
    speakers_importance = speakers_importance.sort_values(by='importance_score', ascending=False)
    
    return speakers_importance

# 分析講者參與度隨時間的變化
def analyze_speaker_participation_over_time(sessions_df, speakers_df):
    # 建立年份到講者的映射
    year_to_speakers = defaultdict(set)
    for _, row in sessions_df.iterrows():
        year = row['year']
        
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
        for speaker in speakers:
            year_to_speakers[year].add(speaker)
    
    # 計算每年新加入的講者和持續參與的講者
    years = sorted(year_to_speakers.keys())
    new_speakers = {}
    continuing_speakers = {}
    
    all_previous_speakers = set()
    for year in years:
        # 今年的講者
        current_speakers = year_to_speakers[year]
        
        # 新加入的講者(首次出現)
        new_speakers[year] = current_speakers - all_previous_speakers
        
        # 持續參與的講者(之前曾出現)
        continuing_speakers[year] = current_speakers & all_previous_speakers
        
        # 更新所有歷史講者集合
        all_previous_speakers.update(current_speakers)
    
    # 建立數據框
    speaker_trends = pd.DataFrame({
        'year': years,
        'total_speakers': [len(year_to_speakers[year]) for year in years],
        'new_speakers': [len(new_speakers[year]) for year in years],
        'continuing_speakers': [len(continuing_speakers[year]) for year in years]
    })
    
    # 繪製趨勢圖
    plt.figure(figsize=(12, 6))
    
    sns.lineplot(data=speaker_trends, x='year', y='total_speakers', marker='o', label='總講者數')
    sns.lineplot(data=speaker_trends, x='year', y='new_speakers', marker='s', label='新講者')
    sns.lineplot(data=speaker_trends, x='year', y='continuing_speakers', marker='^', label='持續參與講者')
    
    plt.title('COSCUP 講者參與度隨時間變化 (2020-2024)')
    plt.xlabel('年份')
    plt.ylabel('講者數量')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('speaker_trends.png', dpi=300)
    plt.close()
    
    return speaker_trends

# 主函數
def main():
    # 載入處理過的資料
    sessions_df = pd.read_csv('data/processed/sessions.csv')
    speakers_df = pd.read_csv('data/processed/speakers.csv')
    
    # 印出資料欄位名稱以供檢查
    print("議程資料欄位：", sessions_df.columns.tolist())
    print("講者資料欄位：", speakers_df.columns.tolist())
    
    # 檢查資料的前幾筆記錄
    print("\n議程資料預覽：")
    print(sessions_df.head())
    print("\n講者資料預覽：")
    print(speakers_df.head())
    
    # 建立講者網絡
    G, speaker_to_sessions = build_speaker_network(sessions_df, speakers_df)
    
    # 社群檢測
    partition, community_counts, main_speakers = detect_communities(G)
    
    # 繪製講者網絡圖
    plot_speaker_network(G, partition)
    
    # 分析重要講者
    key_speakers = analyze_key_speakers(G)
    
    # 分析講者參與度隨時間的變化
    speaker_trends = analyze_speaker_participation_over_time(sessions_df, speakers_df)
    
    # 儲存結果
    key_speakers.to_csv('key_speakers.csv', index=False, encoding='utf-8')
    speaker_trends.to_csv('speaker_trends.csv', index=False, encoding='utf-8')
    
    # 輸出社群分析結果
    with open('community_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('社群分析結果:\n\n')
        for comm, count in sorted(community_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f'社群 {comm}: {count} 位講者\n')
            if comm in main_speakers:
                f.write('主要講者:\n')
                for _, name, count in main_speakers[comm]:
                    f.write(f'  - {name}: {count} 場議程\n')
            f.write('\n')
    
    print("講者網絡分析完成，已儲存視覺化結果")

if __name__ == "__main__":
    main()
