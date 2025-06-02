#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
資料預處理腳本: 處理COSCUP議程JSON資料，轉換為標準化CSV格式
"""

import os
import json
import pandas as pd
import re
import logging
from tqdm import tqdm
import argparse

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_json_file(file_path):
    """
    載入JSON檔案並處理可能的編碼問題

    Args:
        file_path: JSON檔案路徑

    Returns:
        dict: 載入的JSON資料
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except UnicodeDecodeError:
        # 嘗試使用其他編碼
        with open(file_path, 'r', encoding='latin-1') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"無法解析JSON檔案 {file_path}: {e}")
        return None


def extract_year_from_filename(file_path):
    """從檔名中提取年份"""
    filename = os.path.basename(file_path)
    match = re.search(r'coscup_(\d{4})_', filename)
    if match:
        return match.group(1)
    return None


def normalize_session_data(data, year):
    """
    標準化議程資料

    Args:
        data: 原始JSON資料
        year: 資料年份

    Returns:
        tuple: (sessions_df, speakers_df, tracks_df)
    """
    # 檢查資料結構
    if not isinstance(data, dict):
        logger.error(f"無效的資料格式: {type(data)}")
        return None, None, None
    # 處理議程資料
    sessions = []
    if 'sessions' in data:
        for session in data['sessions']:
            # 複製原始資料
            normalized_session = session.copy()
            # 添加年份
            normalized_session['year'] = year
            # 標準化時間格式
            if 'start' in normalized_session:
                try:
                    start_time = pd.to_datetime(normalized_session['start'])
                    normalized_session['start'] = start_time.strftime(
                        '%Y-%m-%dT%H:%M:%S'
                    )
                except ValueError:
                    logger.warning(
                        f"無法解析開始時間: {normalized_session.get('start')}"
                    )
            if 'end' in normalized_session:
                try:
                    end_time = pd.to_datetime(normalized_session['end'])
                    normalized_session['end'] = end_time.strftime(
                        '%Y-%m-%dT%H:%M:%S'
                    )
                except ValueError:
                    logger.warning(
                        f"無法解析結束時間: {normalized_session.get('end')}"
                    )
            # 處理講者欄位 - 確保為列表格式
            if 'speakers' in normalized_session:
                speakers_data = normalized_session['speakers']
                if isinstance(speakers_data, str):
                    try:
                        # 嘗試將字串解析為JSON
                        speakers_list = json.loads(speakers_data)
                        normalized_session['speakers'] = speakers_list
                    except json.JSONDecodeError:
                        # 可能是以逗號分隔的ID列表
                        speakers_list = [
                            s.strip() for s in speakers_data.split(',') if s.strip()
                        ]
                        normalized_session['speakers'] = speakers_list
            # 添加到列表
            sessions.append(normalized_session)
    # 處理講者資料
    speakers = []
    if 'speakers' in data:
        for speaker in data['speakers']:
            # 複製原始資料
            normalized_speaker = speaker.copy()
            # 添加年份
            normalized_speaker['year'] = year
            # 添加到列表
            speakers.append(normalized_speaker)
    # 處理議程軌資料
    tracks = []
    if 'tracks' in data:
        for track in data['tracks']:
            # 複製原始資料
            normalized_track = track.copy()
            # 添加年份
            normalized_track['year'] = year
            # 添加到列表
            tracks.append(normalized_track)
    # 轉換為DataFrame
    sessions_df = pd.DataFrame(sessions) if sessions else pd.DataFrame()
    speakers_df = pd.DataFrame(speakers) if speakers else pd.DataFrame()
    tracks_df = pd.DataFrame(tracks) if tracks else pd.DataFrame()
    return sessions_df, speakers_df, tracks_df


def clean_and_standardize_data(sessions_df, speakers_df, tracks_df):
    """
    清理和標準化資料

    Args:
        sessions_df: 議程DataFrame
        speakers_df: 講者DataFrame
        tracks_df: 議程軌DataFrame

    Returns:
        tuple: (cleaned_sessions_df, cleaned_speakers_df, cleaned_tracks_df)
    """
    # 處理議程資料
    if not sessions_df.empty:
        # 轉換時間欄位
        for col in ['start', 'end']:
            if col in sessions_df.columns:
                sessions_df[col] = pd.to_datetime(
                    sessions_df[col], errors='coerce'
                )
        # 轉換年份為整數
        if 'year' in sessions_df.columns:
            sessions_df['year'] = pd.to_numeric(
                sessions_df['year'], errors='coerce'
            ).astype('Int64')
        # 處理缺失值 - 標量值填充 (speakers 欄位單獨處理)
        columns_to_fill = {
            'title': '',
            'description': '',
            'language': 'unknown',
            'type': 'unknown',
            'room': 'unknown'
        }
        # 只填充 DataFrame 中實際存在的欄位
        fill_values = {
            k: v for k, v in columns_to_fill.items() if k in sessions_df.columns
        }
        if fill_values:
            sessions_df = sessions_df.fillna(value=fill_values)
        # 處理 speakers 欄位：確保其存在並將 NaN/None 值轉換為空列表
        if 'speakers' in sessions_df.columns:
            sessions_df['speakers'] = sessions_df['speakers'].apply(
                lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else x)
            )
        elif not sessions_df.empty:  # 如果欄位不存在且 DataFrame 有資料列
            sessions_df['speakers'] = [[] for _ in range(len(sessions_df))]
        # 如果 DataFrame 為空且 'speakers' 欄位不存在，則不進行操作
    # 處理講者資料
    if not speakers_df.empty:
        # 轉換年份為整數
        if 'year' in speakers_df.columns:
            speakers_df['year'] = pd.to_numeric(
                speakers_df['year'], errors='coerce'
            ).astype('Int64')
        # 處理缺失值
        speakers_df = speakers_df.fillna({
            'name': '',
            'bio': '',
            'company': '',
            'avatar': ''
        })
    # 處理議程軌資料
    if not tracks_df.empty:
        # 轉換年份為整數
        if 'year' in tracks_df.columns:
            tracks_df['year'] = pd.to_numeric(
                tracks_df['year'], errors='coerce'
            ).astype('Int64')
        # 處理缺失值
        tracks_df = tracks_df.fillna({
            'name': '',
            'description': ''
        })
    return sessions_df, speakers_df, tracks_df


def identify_unique_speakers(all_speakers_df):
    """
    識別跨年度的唯一講者

    Args:
        all_speakers_df: 包含所有年份講者的DataFrame

    Returns:
        DataFrame: 添加了unique_id欄位的講者DataFrame
    """
    if all_speakers_df.empty:
        return all_speakers_df
    # 首先使用ID進行合併
    unique_speakers = all_speakers_df.copy()
    unique_speakers['unique_id'] = unique_speakers['id']
    # 檢查不同ID但名稱相同的講者
    if 'name' in unique_speakers.columns:
        name_groups = unique_speakers.groupby('name')
        # 處理名稱相同但ID不同的情況
        for name, group in name_groups:
            if len(group) > 1 and len(set(group['id'])) > 1:
                # 名稱相同但ID不同，可能是同一人
                logger.info(
                    f"發現可能相同的講者: {name}, IDs: {group['id'].tolist()}"
                )
                # 使用最常見的ID作為唯一ID
                most_common_id = group['id'].value_counts().index[0]
                # 更新唯一ID
                unique_speakers.loc[
                    unique_speakers['name'] == name, 'unique_id'
                ] = most_common_id
    return unique_speakers


def merge_yearly_data(yearly_data):
    """
    合併各年度資料

    Args:
        yearly_data: 包含每年sessions_df, speakers_df, tracks_df的字典

    Returns:
        tuple: (merged_sessions_df, merged_speakers_df, merged_tracks_df)
    """
    # 準備合併用的列表
    all_sessions = []
    all_speakers = []
    all_tracks = []
    # 合併各年度資料
    for data in yearly_data.values():
        sessions_df, speakers_df, tracks_df = data
        if not sessions_df.empty:
            all_sessions.append(sessions_df)
        if not speakers_df.empty:
            all_speakers.append(speakers_df)
        if not tracks_df.empty:
            all_tracks.append(tracks_df)
    # 合併DataFrame
    merged_sessions_df = pd.concat(
        all_sessions, ignore_index=True
    ) if all_sessions else pd.DataFrame()
    merged_speakers_df = pd.concat(
        all_speakers, ignore_index=True
    ) if all_speakers else pd.DataFrame()
    merged_tracks_df = pd.concat(
        all_tracks, ignore_index=True
    ) if all_tracks else pd.DataFrame()
    # 處理講者去重
    if not merged_speakers_df.empty:
        # 識別唯一講者
        merged_speakers_df = identify_unique_speakers(merged_speakers_df)
        # 去除重複講者 (保留最新年份的資料)
        if 'unique_id' in merged_speakers_df.columns and \
           'year' in merged_speakers_df.columns:
            # 按唯一ID分組，保留年份最大的記錄
            merged_speakers_df = merged_speakers_df.sort_values(
                'year', ascending=False
            )
            merged_speakers_df = merged_speakers_df.drop_duplicates(
                subset=['unique_id'], keep='first'
            )
    return merged_sessions_df, merged_speakers_df, merged_tracks_df


def extract_session_features(sessions_df):
    """
    提取額外的議程特徵

    Args:
        sessions_df: 議程DataFrame

    Returns:
        DataFrame: 添加了額外特徵的議程DataFrame
    """
    if sessions_df.empty:
        return sessions_df
    # 複製DataFrame以避免修改原始資料
    enhanced_df = sessions_df.copy()
    # 提取時間特徵
    if 'start' in enhanced_df.columns:
        # 確保start是datetime類型
        if not pd.api.types.is_datetime64_dtype(enhanced_df['start']):
            enhanced_df['start'] = pd.to_datetime(
                enhanced_df['start'], errors='coerce'
            )
        # 提取小時、星期幾等
        enhanced_df['hour'] = enhanced_df['start'].dt.hour
        enhanced_df['day_of_week'] = enhanced_df['start'].dt.dayofweek
        enhanced_df['month'] = enhanced_df['start'].dt.month
        enhanced_df['day'] = enhanced_df['start'].dt.day
    # 計算議程時長
    if 'start' in enhanced_df.columns and 'end' in enhanced_df.columns:
        # 確保end是datetime類型
        if not pd.api.types.is_datetime64_dtype(enhanced_df['end']):
            enhanced_df['end'] = pd.to_datetime(
                enhanced_df['end'], errors='coerce'
            )
        # 計算時長（分鐘）
        enhanced_df['duration_minutes'] = \
            (enhanced_df['end'] - enhanced_df['start']).dt.total_seconds() / 60
        # 分類時長
        enhanced_df['duration_category'] = pd.cut(
            enhanced_df['duration_minutes'],
            bins=[0, 30, 60, float('inf')],
            labels=['短講 (≤30分鐘)', '標準 (31-60分鐘)', '長講 (>60分鐘)']
        )
    # 提取標題長度特徵
    if 'title' in enhanced_df.columns:
        enhanced_df['title_length'] = \
            enhanced_df['title'].fillna('').str.len()
    # 提取描述長度特徵
    if 'description' in enhanced_df.columns:
        enhanced_df['description_length'] = \
            enhanced_df['description'].fillna('').str.len()
    # 提取講者數量特徵
    if 'speakers' in enhanced_df.columns:
        # 處理不同格式的speakers欄位
        def count_speakers(speakers_data):
            if isinstance(speakers_data, list):
                return len(speakers_data)
            elif isinstance(speakers_data, str):
                try:
                    return len(json.loads(speakers_data))
                except json.JSONDecodeError:
                    return len(
                        [s.strip() for s in speakers_data.split(',') if s.strip()]
                    )
            else:
                return 0
        enhanced_df['speaker_count'] = \
            enhanced_df['speakers'].apply(count_speakers)
    return enhanced_df


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='預處理COSCUP議程資料')
    parser.add_argument('--input_dir', default='data/raw',
                        help='原始JSON檔案所在目錄')
    parser.add_argument('--output_dir', default='data/processed',
                        help='輸出CSV檔案目錄')
    args = parser.parse_args()
    # 確保輸出目錄存在
    os.makedirs(args.output_dir, exist_ok=True)
    # 讀取資料檔案
    json_files = [f for f in os.listdir(args.input_dir)
                  if f.startswith('coscup_') and f.endswith('_session.json')]
    if not json_files:
        logger.error(f"在 {args.input_dir} 中未找到符合格式的JSON檔案")
        return
    # 按年份處理資料
    yearly_data = {}
    for filename in tqdm(json_files, desc="處理JSON檔案"):
        file_path = os.path.join(args.input_dir, filename)
        # 從檔名提取年份
        year = extract_year_from_filename(file_path)
        if not year:
            logger.warning(f"無法從檔名提取年份: {filename}")
            continue
        # 載入JSON資料
        data = load_json_file(file_path)
        if not data:
            logger.error(f"無法載入JSON檔案: {file_path}")
            continue
        # 標準化資料
        sessions_df, speakers_df, tracks_df = normalize_session_data(data, year)
        # 清理和標準化資料
        sessions_df, speakers_df, tracks_df = clean_and_standardize_data(
            sessions_df, speakers_df, tracks_df)
        # 存儲年度資料
        yearly_data[year] = (sessions_df, speakers_df, tracks_df)
        # 保存單獨年份的資料
        if not sessions_df.empty:
            sessions_df.to_csv(
                f"{args.output_dir}/sessions_{year}.csv",
                index=False, encoding='utf-8'
            )
            logger.info(
                f"已保存 {year} 年議程資料: {len(sessions_df)} 筆記錄"
            )
        if not speakers_df.empty:
            speakers_df.to_csv(
                f"{args.output_dir}/speakers_{year}.csv",
                index=False, encoding='utf-8'
            )
            logger.info(
                f"已保存 {year} 年講者資料: {len(speakers_df)} 筆記錄"
            )
        if not tracks_df.empty:
            tracks_df.to_csv(
                f"{args.output_dir}/tracks_{year}.csv",
                index=False, encoding='utf-8'
            )
            logger.info(
                f"已保存 {year} 年議程軌資料: {len(tracks_df)} 筆記錄"
            )
    # 合併所有年份的資料
    merged_sessions_df, merged_speakers_df, merged_tracks_df = \
        merge_yearly_data(yearly_data)
    # 提取額外特徵
    enhanced_sessions_df = extract_session_features(merged_sessions_df)
    # 保存合併後的資料
    if not enhanced_sessions_df.empty:
        enhanced_sessions_df.to_csv(
            f"{args.output_dir}/sessions.csv", index=False, encoding='utf-8'
        )
        logger.info(
            f"已保存合併後的議程資料: {len(enhanced_sessions_df)} 筆記錄"
        )
    if not merged_speakers_df.empty:
        merged_speakers_df.to_csv(
            f"{args.output_dir}/speakers.csv", index=False, encoding='utf-8'
        )
        logger.info(
            f"已保存合併後的講者資料: {len(merged_speakers_df)} 筆記錄"
        )
    if not merged_tracks_df.empty:
        merged_tracks_df.to_csv(
            f"{args.output_dir}/tracks.csv", index=False, encoding='utf-8'
        )
        logger.info(
            f"已保存合併後的議程軌資料: {len(merged_tracks_df)} 筆記錄"
        )
    # 輸出資料概況
    if not yearly_data:
        logger.warning("沒有處理任何資料")
    else:
        logger.info(f"成功處理了 {len(yearly_data)} 個年份的資料")
        # 統計各年資料量
        year_stats = {}
        for year, (sessions, speakers, tracks) in yearly_data.items():
            year_stats[year] = {
                'sessions': len(sessions) if not sessions.empty else 0,
                'speakers': len(speakers) if not speakers.empty else 0,
                'tracks': len(tracks) if not tracks.empty else 0
            }
        # 轉為DataFrame並保存
        stats_df = pd.DataFrame.from_dict(year_stats, orient='index')
        stats_df.index.name = 'year'
        stats_df.to_csv(
            f"{args.output_dir}/yearly_stats.csv", encoding='utf-8'
        )
        logger.info("已保存年度統計資料")
        # 顯示統計數據
        print("\n各年份資料統計:")
        print(stats_df)
    logger.info("資料預處理完成！")


if __name__ == "__main__":
    main()
