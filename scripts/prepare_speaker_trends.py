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
                
                logging.info(f"正在處理 {year} 年的資料檔案: {filename}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # 確保資料包含必要的欄位
                if not isinstance(raw_data, dict):
                    logging.warning(f"警告：{year} 年資料格式不正確，應為 dict。跳過此檔案。")
                    continue
                    
                # 重組資料結構
                yearly_data[year] = {
                    "sessions": [],
                    "speakers": [],
                    "session_types": []
                }
                
                # 處理議程類型
                if "session_types" in raw_data and isinstance(raw_data["session_types"], list):
                    for st_raw in raw_data["session_types"]:
                        if isinstance(st_raw, dict):
                            st_id = st_raw.get("id")
                            st_zh_name = st_raw.get("zh", {}).get("name")
                            if st_id and st_zh_name:
                                yearly_data[year]["session_types"].append({
                                    "id": st_id,
                                    "zh": {"name": st_zh_name}
                                })
                
                # 收集所有實際參與議程的講者 ID，並處理議程資料 (sessions)
                speaker_ids_from_sessions = set()
                if "sessions" in raw_data and isinstance(raw_data["sessions"], list):
                    for session_raw in raw_data["sessions"]:
                        if not isinstance(session_raw, dict):
                            continue

                        current_session_speaker_ids_processed = []
                        raw_session_speakers_field = session_raw.get("speakers")

                        # 處理 session_raw["speakers"] 可能的格式 (字串JSON, 字串逗號分隔, 列表)
                        if isinstance(raw_session_speakers_field, str):
                            try:
                                parsed_speakers = json.loads(raw_session_speakers_field)
                                if isinstance(parsed_speakers, list):
                                    current_session_speaker_ids_processed.extend(str(sid).strip() for sid in parsed_speakers if str(sid).strip())
                            except json.JSONDecodeError:
                                current_session_speaker_ids_processed.extend(s.strip() for s in raw_session_speakers_field.split(',') if s.strip())
                        elif isinstance(raw_session_speakers_field, list):
                            current_session_speaker_ids_processed.extend(str(sid).strip() for sid in raw_session_speakers_field if str(sid).strip())
                        
                        session_zh_title = session_raw.get("zh", {}).get("title")
                        session_type_id = session_raw.get("type")

                        if session_zh_title and current_session_speaker_ids_processed:
                            yearly_data[year]["sessions"].append({
                                "speakers": current_session_speaker_ids_processed,
                                "type": session_type_id, # type 可以是 None
                                "zh": {"title": session_zh_title}
                            })
                            speaker_ids_from_sessions.update(current_session_speaker_ids_processed)
                
                # 處理講者資料 (speakers)
                if "speakers" in raw_data and isinstance(raw_data["speakers"], list):
                    for spk_raw in raw_data["speakers"]:
                        if isinstance(spk_raw, dict):
                            spk_id = str(spk_raw.get("id", "")).strip()
                            spk_zh_name = spk_raw.get("zh", {}).get("name")
                            if spk_id and spk_zh_name and spk_id in speaker_ids_from_sessions:
                                yearly_data[year]["speakers"].append({
                                    "id": spk_id,
                                    "zh": {"name": spk_zh_name}
                                })
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
    output_path = output_dir / "speakerTrends.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(yearly_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"資料已處理完成並儲存至：{output_path}")
    
    # 印出基本統計資訊
    for year, data in yearly_data.items():
        logging.info(f"\n{year} 年度統計：")
        logging.info(f"  - 議程數量 (Sessions): {len(data.get('sessions', []))}")
        logging.info(f"  - 講者數量 (Speakers): {len(data.get('speakers', []))}")
        logging.info(f"  - 議程類型數量 (Session Types): {len(data.get('session_types', []))}")

if __name__ == "__main__":
    main()