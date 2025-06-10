import json
import os
from pathlib import Path
import logging

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_yearly_data(raw_dir):
    """載入各年度的原始資料並轉換成所需格式"""
    yearly_data = {}
    
    try:
        # 讀取所有年度的 JSON 檔案
        for filename in sorted(os.listdir(raw_dir)):
            if filename.startswith('coscup_') and filename.endswith('_session.json'):
                year = filename.split('_')[1]
                file_path = Path(raw_dir) / filename
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # 確保資料包含必要的欄位
                if not isinstance(raw_data, dict):
                    print(f"警告：{year} 年資料格式不正確")
                    continue
                    
                # 重組資料結構
                yearly_data[year] = {
                    "sessions": [],     # 議程資料
                    "speakers": [],     # 講者資料
                    "session_types": [] # 議程類型
                }
                
                # 處理議程類型
                if "session_types" in raw_data:
                    yearly_data[year]["session_types"] = raw_data["session_types"]
                
                # 處理議程和講者資料
                if "sessions" in raw_data:
                    yearly_data[year]["sessions"] = raw_data["sessions"]
                    
                    # 收集所有講者 ID
                    speakers = set()
                    for session in raw_data["sessions"]:
                        if "speakers" in session:
                            # 處理可能的不同格式
                            session_speakers = session["speakers"]
                            if isinstance(session_speakers, str):
                                # 如果是字串，嘗試解析 JSON
                                try:
                                    speaker_list = json.loads(session_speakers)
                                    speakers.update(speaker_list)
                                except json.JSONDecodeError:
                                    # 如果不是 JSON，假設是逗號分隔的字串
                                    speakers.update(s.strip() for s in session_speakers.split(','))
                            elif isinstance(session_speakers, list):
                                # 如果是列表，直接加入
                                speakers.update(session_speakers)
                    
                    # 加入講者資料
                    if "speakers" in raw_data:
                        yearly_data[year]["speakers"] = [
                            speaker for speaker in raw_data["speakers"]
                            if isinstance(speaker, dict) and speaker.get("id") in speakers
                        ]
    except Exception as e:
        logging.error(f"處理資料時發生錯誤：{str(e)}")
        raise
    
    return yearly_data

def main():
    # 設定路徑
    raw_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入並處理資料
    yearly_data = load_yearly_data(raw_dir)
    
    # 儲存處理後的資料
    output_path = output_dir / "speaker_trends_input.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(yearly_data, f, ensure_ascii=False, indent=2)
    
    print(f"資料已處理完成並儲存至：{output_path}")
    
    # 印出基本統計資訊
    for year, data in yearly_data.items():
        print(f"\n{year} 年度統計：")
        print(f"- 議程數量：{len(data['sessions'])}")
        print(f"- 講者數量：{len(data['speakers'])}")
        print(f"- 議程類型數量：{len(data['session_types'])}")

if __name__ == "__main__":
    main()