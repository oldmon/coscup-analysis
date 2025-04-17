from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import os
import networkx as nx
from collections import defaultdict

app = Flask(__name__, 
            static_folder='dashboard/static',
            template_folder='dashboard/templates')

# 載入處理好的資料
def load_data():
    data = {}
    
    # 檢查並載入各類資料檔案
    data_files = {
        'sessions': 'data/processed/sessions.csv',
        'speakers': 'data/processed/speakers.csv',
        'tracks': 'data/processed/tracks.csv',
        'topic_trends': 'data/processed/yearly_topic_trends.csv',
        'session_topics': 'data/processed/session_topics.csv',
        'key_speakers': 'data/processed/key_speakers.csv',
        'speaker_trends': 'data/processed/speaker_trends.csv',
        'conference_growth': 'data/processed/conference_growth.csv',
    }
    
    for key, file_path in data_files.items():
        if os.path.exists(file_path):
            data[key] = pd.read_csv(file_path)
        else:
            print(f"Warning: {file_path} not found")
    
    return data

# 全域資料
DATA = load_data()

@app.route('/')
def index():
    """主頁：渲染儀表板"""
    return render_template('dashboard.html')

@app.route('/api/conference_overview')
def conference_overview():
    """會議概況資料"""
    if 'conference_growth' in DATA:
        df = DATA['conference_growth']
        result = {
            'years': df['year'].tolist(),
            'sessions': df['sessions'].tolist(),
            'speakers': df['speakers'].tolist() if 'speakers' in df.columns else [],
            'tracks': df['tracks'].tolist() if 'tracks' in df.columns else []
        }
        return jsonify(result)
    return jsonify({'error': 'Data not available'})

@app.route('/api/language_distribution')
def language_distribution():
    """議程語言分布資料"""
    if 'sessions' in DATA:
        df = DATA['sessions']
        if 'language' in df.columns and 'year' in df.columns:
            # 計算每年各語言的數量
            lang_counts = df.groupby(['year', 'language']).size().unstack(fill_value=0)
            
            # 計算各語言占比
            lang_percent = lang_counts.div(lang_counts.sum(axis=1), axis=0) * 100
            
            result = {
                'years': lang_percent.index.tolist(),
            }
            
            # 添加各語言的數據
            for lang in lang_percent.columns:
                result[lang] = lang_percent[lang].tolist()
                
            return jsonify(result)
    return jsonify({'error': 'Language data not available'})

@app.route('/api/topic_keywords')
def topic_keywords():
    """主題關鍵詞資料"""
    # 這裡可能需要從特定文件載入，如果沒有現成的資料可以使用TF-IDF動態計算
    # 示例返回值
    result = {}
    
    if 'session_topics' in DATA:
        # 根據主題模型結果提取關鍵詞
        df = DATA['session_topics']
        if 'main_topic' in df.columns and 'year' in df.columns:
            years = df['year'].unique()
            
            for year in years:
                year_df = df[df['year'] == year]
                # 假設我們有一個all_topics列包含主題分佈
                # 提取每年的熱門主題及其權重
                result[str(year)] = {}
                
                # 這裡應該根據實際資料結構調整
                # 示例: 計算每個主題的出現次數
                topic_counts = year_df['main_topic'].value_counts().head(10)
                for topic, count in topic_counts.items():
                    result[str(year)][f'Topic {topic}'] = int(count)
    
    # 如果沒有資料，返回示例資料
    if not result:
        result = {
            '2020': {'AI': 25, 'Cloud': 22, 'DevOps': 20, 'Security': 18},
            '2021': {'Cloud': 28, 'DevOps': 25, 'AI': 22, 'Security': 20},
            '2022': {'AI': 30, 'Cloud': 28, 'Security': 25, 'DevOps': 22},
            '2023': {'AI': 35, 'Security': 30, 'Cloud': 25, 'DevOps': 20},
            '2024': {'AI': 40, 'Security': 35, 'Cloud': 30, 'DevOps': 25}
        }
    
    return jsonify(result)

@app.route('/api/topic_trends')
def topic_trends():
    """主題趨勢變化資料"""
    if 'topic_trends' in DATA:
        df = DATA['topic_trends']
        result = {'years': df.index.tolist()}
        
        # 添加各主題的趨勢數據
        for column in df.columns:
            result[column] = df[column].tolist()
            
        return jsonify(result)
    
    # 示例資料
    result = {
        'years': ['2020', '2021', '2022', '2023', '2024'],
        'Topic 0': [25, 28, 30, 32, 35],
        'Topic 1': [20, 22, 25, 28, 30],
        'Topic 2': [15, 18, 20, 22, 25],
        'Topic 3': [10, 12, 15, 18, 20],
        'Topic 4': [30, 28, 25, 22, 20]
    }
    
    return jsonify(result)

@app.route('/api/speaker_network')
def speaker_network():
    """講者合作網絡資料"""
    year = request.args.get('year', 'all')
    
    if 'sessions' in DATA and 'speakers' in DATA and 'key_speakers' in DATA:
        sessions_df = DATA['sessions']
        speakers_df = DATA['speakers']
        key_speakers_df = DATA['key_speakers']
        
        # 篩選特定年份的資料
        if year != 'all' and 'year' in sessions_df.columns:
            sessions_df = sessions_df[sessions_df['year'] == year]
        
        # 構建網絡 (這裡需要根據實際資料結構調整)
        G = nx.Graph()
        
        # 添加節點
        for _, speaker in key_speakers_df.iterrows():
            G.add_node(speaker['speaker_id'], 
                    name=speaker['name'],
                    company=speaker.get('company', ''),
                    community=speaker.get('community', 0),
                    importance=speaker.get('importance_score', 0.5))
        
        # 添加邊 (假設我們知道如何從sessions中提取講者合作關係)
        # 這部分需要根據實際資料結構調整
        
        # 將網絡轉換為JSON格式
        nodes = [{'id': n, 
                'name': G.nodes[n].get('name', f'Speaker {n}'),
                'community': G.nodes[n].get('community', 0),
                'size': 10 + G.nodes[n].get('importance', 0.5) * 20} 
                for n in G.nodes()]
        
        links = [{'source': u, 
                'target': v, 
                'value': G.edges[u, v].get('weight', 1)} 
                for u, v in G.edges()]
        
        return jsonify({'nodes': nodes, 'links': links})
    
    # 示例資料
    # 生成一些隨機節點和連接
    import random
    random.seed(42)
    
    nodes = []
    links = []
    communities = [0, 1, 2, 3, 4]
    
    # 生成節點
    for i in range(1, 31):
        community = random.choice(communities)
        importance = random.random() * 0.5 + 0.5  # 0.5 - 1.0
        
        nodes.append({
            'id': f'speaker_{i}',
            'name': f'講者 {i}',
            'community': community,
            'size': 10 + importance * 20
        })
    
    # 生成連接
    for i in range(len(nodes)):
        # 每個節點與1-3個其他節點連接
        for _ in range(random.randint(1, 3)):
            target = random.randint(0, len(nodes) - 1)
            if i != target:
                links.append({
                    'source': nodes[i]['id'],
                    'target': nodes[target]['id'],
                    'value': random.randint(1, 3)
                })
    
    return jsonify({'nodes': nodes, 'links': links})

@app.route('/api/time_distribution')
def time_distribution():
    """議程時間分布資料"""
    if 'sessions' in DATA:
        df = DATA['sessions']
        if 'start' in df.columns and 'year' in df.columns:
            # 轉換時間格式並提取小時
            df['start_datetime'] = pd.to_datetime(df['start'])
            df['hour'] = df['start_datetime'].dt.hour
            
            # 計算每年每小時的議程數量
            hour_counts = df.groupby(['year', 'hour']).size().unstack(fill_value=0)
            
            result = {
                'hours': [str(h) for h in range(9, 18)],  # 假設9:00-18:00
            }
            
            for year in hour_counts.index:
                result[str(year)] = []
                for hour in range(9, 18):
                    if hour in hour_counts.columns:
                        result[str(year)].append(int(hour_counts.loc[year, hour]))
                    else:
                        result[str(year)].append(0)
            
            return jsonify(result)
    
    # 示例資料
    result = {
        'hours': ['9', '10', '11', '12', '13', '14', '15', '16', '17'],
        '2020': [5, 10, 15, 5, 10, 15, 10, 5, 0],
        '2021': [8, 12, 18, 8, 12, 18, 12, 8, 0],
        '2022': [10, 15, 20, 10, 15, 20, 15, 10, 5],
        '2023': [12, 18, 25, 12, 18, 25, 18, 12, 8],
        '2024': [15, 20, 30, 15, 20, 30, 20, 15, 10]
    }
    
    return jsonify(result)

@app.route('/api/speaker_trends')
def speaker_trends():
    """講者參與趨勢資料"""
    if 'speaker_trends' in DATA:
        df = DATA['speaker_trends']
        result = {
            'years': df['year'].tolist(),
            'total_speakers': df['total_speakers'].tolist(),
            'new_speakers': df['new_speakers'].tolist(),
            'continuing_speakers': df['continuing_speakers'].tolist()
        }
        return jsonify(result)
    
    # 示例資料
    result = {
        'years': ['2020', '2021', '2022', '2023', '2024'],
        'total_speakers': [80, 100, 120, 140, 160],
        'new_speakers': [80, 60, 50, 60, 70],
        'continuing_speakers': [0, 40, 70, 80, 90]
    }
    
    return jsonify(result)

@app.route('/api/community_analysis')
def community_analysis():
    """社群分析資料"""
    if 'key_speakers' in DATA:
        df = DATA['key_speakers']
        if 'community' in df.columns:
            # 計算每個社群的講者數量
            community_counts = df['community'].value_counts().to_dict()
            
            # 計算每個社群的主要公司/組織
            communities = {}
            for comm, count in community_counts.items():
                comm_df = df[df['community'] == comm]
                # 假設有company欄位
                if 'company' in comm_df.columns:
                    top_companies = comm_df['company'].value_counts().head(3).to_dict()
                    communities[int(comm)] = {
                        'count': int(count),
                        'top_companies': [{'name': company, 'count': int(comp_count)} 
                                        for company, comp_count in top_companies.items()]
                    }
            
            return jsonify(communities)
    
    # 示例資料
    communities = {
        0: {
            'count': 25,
            'top_companies': [
                {'name': '開源基金會', 'count': 10},
                {'name': '科技公司', 'count': 8},
                {'name': '其他', 'count': 7}
            ]
        },
        1: {
            'count': 20,
            'top_companies': [
                {'name': '科技公司', 'count': 9},
                {'name': '軟體公司', 'count': 6},
                {'name': '其他', 'count': 5}
            ]
        },
        2: {
            'count': 18,
            'top_companies': [
                {'name': '雲端服務商', 'count': 8},
                {'name': '資安公司', 'count': 6},
                {'name': '其他', 'count': 4}
            ]
        },
        3: {
            'count': 15,
            'top_companies': [
                {'name': '開源基金會', 'count': 7},
                {'name': '軟體公司', 'count': 5},
                {'name': '其他', 'count': 3}
            ]
        },
        4: {
            'count': 12,
            'top_companies': [
                {'name': '資安公司', 'count': 5},
                {'name': '科技公司', 'count': 4},
                {'name': '其他', 'count': 3}
            ]
        }
    }
    
    return jsonify(communities)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
