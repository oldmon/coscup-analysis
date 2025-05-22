#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
資料收集腳本: 下載並儲存COSCUP 2020-2024年度議程資料
"""

import os
import json
import requests
from tqdm import tqdm
import pandas as pd
import logging
import time
from urllib.parse import urlparse
import argparse

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """建立必要的目錄結構"""
    dirs = ['data/raw', 'data/processed', 
            'visualizations/static', 'visualizations/interactive']
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"確認目錄存在: {dir_path}")

def download_file(url, save_path, retry=3, timeout=30):
    """
    下載檔案並保存
    
    Args:
        url: 檔案URL
        save_path: 保存路徑
        retry: 重試次數
        timeout: 請求超時時間(秒)
    
    Returns:
        bool: 下載是否成功
    """
    for attempt in range(retry):
        try:
            # 發送請求
            logger.info(f"下載檔案: {url} (嘗試 {attempt+1}/{retry})")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # 檢查回應狀態
            
            # 保存檔案
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"成功保存到: {save_path}")
            return True
            
        except (requests.RequestException, IOError) as e:
            logger.warning(f"下載失敗: {e}")
            if attempt < retry - 1:
                wait_time = 2 ** attempt  # 指數退避
                logger.info(f"等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
    
    logger.error(f"無法下載 {url} 經過 {retry} 次嘗試")
    return False

def download_coscup_data(years=None):
    """
    下載指定年份的COSCUP議程資料
    
    Args:
        years: 要下載的年份列表，None表示下載所有年份(2020-2024)
    
    Returns:
        dict: 下載結果統計
    """
    if years is None:
        years = range(2020, 2025)  # 2020-2024
    
    results = {'success': 0, 'failed': 0, 'skipped': 0}
    
    for year in tqdm(years, desc="下載年度資料"):
        url = f"https://coscup.org/{year}/json/session.json"
        save_path = f"data/raw/coscup_{year}_session.json"
        
        # 檢查檔案是否已存在
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            if file_size > 100:  # 檔案大小 > 100 bytes，假設內容完整
                logger.info(f"檔案已存在，跳過下載: {save_path} ({file_size} bytes)")
                results['skipped'] += 1
                continue
        
        # 下載檔案
        success = download_file(url, save_path)
        if success:
            results['success'] += 1
        else:
            results['failed'] += 1
    
    return results

def validate_json_files():
    """
    驗證下載的JSON檔案格式是否正確
    
    Returns:
        list: 有效的JSON檔案路徑列表
    """
    valid_files = []
    
    json_files = [f for f in os.listdir('data/raw') 
                 if f.startswith('coscup_') and f.endswith('_session.json')]
    
    for filename in tqdm(json_files, desc="驗證JSON檔案"):
        file_path = os.path.join('data/raw', filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 基本檢查 - 確保檔案包含預期的資料結構
            if isinstance(data, dict):
                valid_files.append(file_path)
                logger.info(f"有效的JSON檔案: {file_path}")
            else:
                logger.warning(f"JSON檔案格式不正確: {file_path}")
        
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"無法解析JSON檔案 {file_path}: {e}")
    
    return valid_files

def save_file_info(valid_files):
    """
    保存有效檔案的基本信息
    
    Args:
        valid_files: 有效檔案路徑列表
    """
    file_info = []
    
    for file_path in valid_files:
        filename = os.path.basename(file_path)
        year = filename.split('_')[1]  # 從檔名提取年份
        
        # 獲取檔案大小
        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes / 1024
        
        # 讀取檔案內容進行基本分析
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 獲取資料結構資訊
            keys = list(data.keys())
            
            # 計算議程數量 (如果存在)
            session_count = len(data.get('sessions', []))
            
            # 計算講者數量 (如果存在)
            speaker_count = len(data.get('speakers', []))
            
            # 計算議程軌數量 (如果存在)
            track_count = len(data.get('tracks', []))
        
        file_info.append({
            'year': year,
            'file_name': filename,
            'file_path': file_path,
            'file_size_kb': round(size_kb, 2),
            'keys': keys,
            'session_count': session_count,
            'speaker_count': speaker_count,
            'track_count': track_count
        })
    
    # 保存為CSV
    pd.DataFrame(file_info).to_csv('data/raw/file_info.csv', index=False, encoding='utf-8')
    logger.info(f"已將檔案資訊保存到 data/raw/file_info.csv")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='下載COSCUP議程資料')
    parser.add_argument('--years', type=int, nargs='+',
                        help='要下載的年份，如: 2020 2021 2022')
    args = parser.parse_args()
    
    # 設置目錄
    setup_directories()
    
    # 下載資料
    if args.years:
        years = args.years
    else:
        years = None
    
    results = download_coscup_data(years)
    logger.info(f"下載結果: 成功={results['success']}, "
                f"已跳過={results['skipped']}, 失敗={results['failed']}")
    
    # 驗證檔案
    valid_files = validate_json_files()
    logger.info(f"找到 {len(valid_files)} 個有效的JSON檔案")
    
    # 保存檔案資訊
    save_file_info(valid_files)
    
    logger.info("資料收集完成！")

if __name__ == "__main__":
    main()
