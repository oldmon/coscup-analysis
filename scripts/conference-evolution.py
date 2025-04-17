import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from collections import Counter, defaultdict
import networkx as nx
import calendar

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 分析議程軌(Tracks)的演變
def analyze_tracks_evolution(sessions_df, tracks_df):
    # 計算每年各議程軌的議程數量
    track_counts = sessions_df.groupby(['year', 'track']).size().unstack(fill_value=0)
    
    # 計算每年議程軌的總數
    yearly_track_counts = tracks_df.groupby('year').size()
    
    # 繪製議程軌數量隨時間變化圖
    plt.figure(figsize=(10, 6))
    yearly_track_counts.plot(kind='bar', color='skyblue')
    plt.title('COSCUP 議程軌數量變化 (2020-2024)')
    plt.xlabel('年份')
    plt.ylabel('議程軌數量')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('track_count_by_year.png', dpi=300)
    plt.close()
    
    # 繪製熱力圖，顯示每年各議程軌的議程數量
    plt.figure(figsize=(14, 10))
    sns.heatmap(track_counts, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('COSCUP 各議程軌議程數量 (2020-2024)')
    plt.tight_layout()
    plt.savefig('track_session_heatmap.png', dpi=300)
    plt.close()
    
    return track_counts, yearly_track_counts

# 分析議程類型和時長
def analyze_session_types_and_duration(sessions_df):
    # 確保有正確的 start 和 end 時間列
    if 'start' in sessions_df.columns and 'end' in sessions_df.columns:
        # 轉換時間格式
        sessions_df['start_time'] = pd.to_datetime(sessions_df['start'])
        sessions_df['end_time'] = pd.to_datetime(sessions_df['end'])
        
        # 計算時長（分鐘）
        sessions_df['duration_minutes'] = (sessions_df['end_time'] - sessions_df['start_time']).dt.total_seconds() / 60
        
        # 根據時長分類
        def categorize_duration(minutes):
            if minutes <= 30:
                return '短講 (≤30分鐘)'
            elif minutes <= 60:
                return '標準 (31-60分鐘)'
            else:
                return '長講 (>60分鐘)'
        
        sessions_df['duration_category'] = sessions_df['duration_minutes'].apply(categorize_duration)
    
    # 分析議程類型
    if 'type' in sessions_df.columns:
        type_counts = sessions_df.groupby(['year', 'type']).size().unstack(fill_value=0)
        
        # 繪製議程類型隨時間變化圖
        plt.figure(figsize=(12, 6))
        type_counts.plot(kind='bar', stacked=True, colormap='tab10')
        plt.title('COSCUP 議程類型分布 (2020-2024)')
        plt.xlabel('年份')
        plt.ylabel('議程數量')
        plt.legend(title='議程類型')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('session_types_by_year.png', dpi=300)
        plt.close()
    
    # 分析議程時長
    if 'duration_category' in sessions_df.columns:
        duration_counts = sessions_df.groupby(['year', 'duration_category']).size().unstack(fill_value=0)
        
        # 繪製議程時長分布圖
        plt.figure(figsize=(12, 6))
        duration_counts.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title('COSCUP 議程時長分布 (2020-2024)')
        plt.xlabel('年份')
        plt.ylabel('議程數量')
        plt.legend(title='議程時長')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('session_duration_by_year.png', dpi=300)
        plt.close()
        
        return sessions_df, type_counts, duration_counts
    
    return sessions_df, None, None

# 分析議程時間分布
def analyze_session_time_distribution(sessions_df):
    # 確保有正確的 start 列
    if 'start' in sessions_df.columns:
        # 轉換時間格式
        sessions_df['start_time'] = pd.to_datetime(sessions_df['start'])
        
        # 提取小時和日期
        sessions_df['hour'] = sessions_df['start_time'].dt.hour
        sessions_df['date'] = sessions_df['start_time'].dt.date
        sessions_df['day_of_week'] = sessions_df['start_time'].dt.dayofweek
        sessions_df['day_name'] = sessions_df['start_time'].dt.day_name()
        
        # 按年份和小時分組計算議程數量
        hourly_counts = sessions_df.groupby(['year', 'hour']).size().unstack(fill_value=0)
        
        # 繪製每小時議程分布熱力圖
        plt.figure(figsize=(14, 8))
        sns.heatmap(hourly_counts, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('COSCUP 議程時間分布 (按小時, 2020-2024)')
        plt.xlabel('小時')
        plt.ylabel('年份')
        plt.tight_layout()
        plt.savefig('session_hourly_distribution.png', dpi=300)
        plt.close()
        
        # 按年份和星期幾分組計算議程數量
        day_counts = sessions_df.groupby(['year', 'day_name']).size().unstack(fill_value=0)
        
        # 繪製每天議程分布條形圖
        plt.figure(figsize=(12, 6))
        day_counts.plot(kind='bar', stacked=True, colormap='Pastel1')
        plt.title('COSCUP 議程星期分布 (2020-2024)')
        plt.xlabel('年份')
        plt.ylabel('議程數量')
        plt.legend(title='星期')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('session_day_distribution.png', dpi=300)
        plt.close()
        
        return hourly_counts, day_counts
    
    return None, None

# 分析議程語言分布
def analyze_session_languages(sessions_df):
    if 'language' in sessions_df.columns:
        # 統計每年各語言的數量
        language_counts = sessions_df.groupby(['year', 'language']).size().unstack(fill_value=0)
        
        # 計算語言比例
        language_percentage = language_counts.div(language_counts.sum(axis=1), axis=0) * 100
        
        # 繪製語言分布趨勢圖
        plt.figure(figsize=(12, 6))
        language_percentage.plot(kind='bar', stacked=True, colormap='Set3')
        plt.title('COSCUP 議程語言分布 (2020-2024)')
        plt.xlabel('年份')
        plt.ylabel('百分比')
        plt.legend(title='語言')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('session_language_distribution.png', dpi=300)
        plt.close()
        
        return language_counts, language_percentage
    
    return None, None

# 分析議程室使用情況
def analyze_room_usage(sessions_df):
    if 'room' in sessions_df.columns:
        # 按年份和議程室分組計算議程數量
        room_counts = sessions_df.groupby(['year', 'room']).size().unstack(fill_value=0)
        
        # 計算每年的議程室數量
        room_numbers = room_counts.gt(0).sum(axis=1)
        
        # 繪製議程室數量變化趨勢
        plt.figure(figsize=(10, 6))
        room_numbers.plot(kind='line', marker='o', color='purple')
        plt.title('COSCUP 議程室數量變化 (2020-2024)')
        plt.xlabel('年份')
        plt.ylabel('議程室數量')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('room_count_trend.png', dpi=300)
        plt.close()
        
        # 繪製議程室使用熱力圖
        plt.figure(figsize=(16, 10))
        sns.heatmap(room_counts, annot=True, fmt='d', cmap='Blues')
        plt.title('COSCUP 議程室使用情況 (2020-2024)')
        plt.xlabel('議程室')
        plt.ylabel('年份')
        plt.tight_layout()
        plt.savefig('room_usage_heatmap.png', dpi=300)
        plt.close()
        
        return room_counts, room_numbers
    
    return None, None

# 分析會議規模變化
def analyze_conference_growth(sessions_df, speakers_df, tracks_df):
    # 計算每年的議程數、講者數和議程軌數
    yearly_sessions = sessions_df.groupby('year').size()
    yearly_speakers = speakers_df.groupby('year').size() if 'year' in speakers_df.columns else None
    yearly_tracks = tracks_df.groupby('year').size() if 'year' in tracks_df.columns else None
    
    # 建立會議規模數據框
    growth_data = pd.DataFrame({
        'sessions': yearly_sessions
    })
    
    if yearly_speakers is not None:
        growth_data['speakers'] = yearly_speakers
        
    if yearly_tracks is not None:
        growth_data['tracks'] = yearly_tracks
    
    # 繪製會議規模趨勢圖
    plt.figure(figsize=(12, 6))
    
    for column in growth_data.columns:
        plt.plot(growth_data.index, growth_data[column], marker='o', label=column)
    
    plt.title('COSCUP 會議規模變化 (2020-2024)')
    plt.xlabel('年份')
    plt.ylabel('數量')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('conference_growth.png', dpi=300)
    plt.close()
    
    return growth_data

# 主函數
def main():
    # 載入處理過的資料
    sessions_df = pd.read_csv('sessions.csv')
    speakers_df = pd.read_csv('speakers.csv')
    tracks_df = pd.read_csv('tracks.csv')
    
    # 分析議程軌的演變
    track_counts, yearly_track_counts = analyze_tracks_evolution(sessions_df, tracks_df)
    
    # 分析議程類型和時長
    sessions_df, type_counts, duration_counts = analyze_session_types_and_duration(sessions_df)
    
    # 分析議程時間分布
    hourly_counts, day_counts = analyze_session_time_distribution(sessions_df)
    
    # 分析議程語言分布
    language_counts, language_percentage = analyze_session_languages(sessions_df)
    
    # 分析議程室使用情況
    room_counts, room_numbers = analyze_room_usage(sessions_df)
    
    # 分析會議規模變化
    growth_data = analyze_conference_growth(sessions_df, speakers_df, tracks_df)
    
    # 儲存分析結果
    growth_data.to_csv('conference_growth.csv', encoding='utf-8')
    
    if type_counts is not None:
        type_counts.to_csv('session_types.csv', encoding='utf-8')
    
    if duration_counts is not None:
        duration_counts.to_csv('session_durations.csv', encoding='utf-8')
    
    if language_percentage is not None:
        language_percentage.to_csv('language_distribution.csv', encoding='utf-8')
    
    print("會議結構與演進分析完成，已儲存視覺化結果")

if __name__ == "__main__":
    main()
