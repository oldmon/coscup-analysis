import json
import pandas as pd
import glob
from datetime import datetime
import os
import re
import requests
from collections import defaultdict

# 下載資料
def download_data():
    years = [2020, 2021, 2022, 2023, 2024]
    for year in years:
        url = f"https://coscup.org/{year}/json/session.json"
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"session_{year}.json", "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"下載 {year} 資料成功")
        else:
            print(f"下載 {year} 資料失敗: {response.status_code}")

# 讀取所有年份的資料
def load_all_data():
    all_data = {}
    for file in glob.glob("session_*.json"):
        year = re.search(r'(\d{4})', file).group(1)
        with open(file, "r", encoding="utf-8") as f:
            all_data[year] = json.load(f)
    return all_data

# 統一資料結構
def normalize_data(all_data):
    # 準備統一的資料結構
    unified_data = {
        "sessions": [],
        "speakers": [],
        "tracks": [],
        "years": []
    }
    
    speaker_ids = set()  # 追蹤已處理的講者ID
    
    for year, data in all_data.items():
        # 假設每年的資料結構可能不同，需針對性處理
        # 這裡只是示例，實際需要根據真實資料結構調整
        
        # 處理議程
        if "sessions" in data:
            for session in data["sessions"]:
                session["year"] = year
                unified_data["sessions"].append(session)
        
        # 處理講者
        if "speakers" in data:
            for speaker in data["speakers"]:
                if speaker["id"] not in speaker_ids:
                    speaker_ids.add(speaker["id"])
                    unified_data["speakers"].append(speaker)
        
        # 處理議程軌
        if "tracks" in data:
            for track in data["tracks"]:
                track["year"] = year
                unified_data["tracks"].append(track)
        
        # 記錄年份資訊
        unified_data["years"].append({
            "year": year,
            "session_count": len(data.get("sessions", [])),
            "speaker_count": len(data.get("speakers", [])),
            "track_count": len(data.get("tracks", []))
        })
    
    return unified_data

# 建立資料分析用的DataFrame
def create_dataframes(unified_data):
    # 議程DataFrame
    sessions_df = pd.DataFrame(unified_data["sessions"])
    
    # 講者DataFrame
    speakers_df = pd.DataFrame(unified_data["speakers"])
    
    # 議程軌DataFrame
    tracks_df = pd.DataFrame(unified_data["tracks"])
    
    # 年度統計DataFrame
    years_df = pd.DataFrame(unified_data["years"])
    
    return {
        "sessions": sessions_df,
        "speakers": speakers_df,
        "tracks": tracks_df,
        "years": years_df
    }

# 執行資料預處理
def main():
    # 下載資料
    download_data()
    
    # 讀取資料
    all_data = load_all_data()
    
    # 統一資料結構
    unified_data = normalize_data(all_data)
    
    # 建立DataFrame
    dataframes = create_dataframes(unified_data)
    
    # 儲存處理後的資料
    for name, df in dataframes.items():
        df.to_csv(f"{name}.csv", index=False, encoding="utf-8")
    
    print("資料預處理完成，已儲存為CSV檔案")

if __name__ == "__main__":
    main()
