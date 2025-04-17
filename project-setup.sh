# 專案需要的Python套件
# 建立requirements.txt檔案
numpy==1.22.4
pandas==1.5.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
networkx==3.1
community==1.0.0b1
jieba==0.42.1
wordcloud==1.8.2.2
gensim==4.3.1
pyLDAvis==3.4.1
requests==2.31.0
tqdm==4.65.0
Flask==2.3.2  # 用於建立簡單的網頁伺服器展示結果

# 專案結構
# 
# coscup-analysis/
# ├── data/                      # 原始及處理後的資料
# │   ├── raw/                   # 原始JSON檔案
# │   └── processed/             # 處理後的CSV檔案
# ├── scripts/                   # 資料處理和分析腳本
# │   ├── 01_data_collection.py  # 資料收集
# │   ├── 02_preprocessing.py    # 資料預處理
# │   ├── 03_topic_analysis.py   # 主題分析
# │   ├── 04_speaker_network.py  # 講者網絡分析
# │   └── 05_conference_stats.py # 會議結構分析
# ├── visualizations/            # 視覺化結果
# │   ├── static/                # 靜態圖表
# │   └── interactive/           # 互動式圖表
# ├── dashboard/                 # 互動式儀表板
# │   ├── static/                # CSS、JS等靜態檔案
# │   └── templates/             # HTML模板
# ├── app.py                     # Flask應用程式進入點
# ├── requirements.txt           # 專案依賴
# └── README.md                  # 專案說明

# 執行步驟腳本 (run.sh)
#!/bin/bash

# 創建目錄結構
mkdir -p data/raw data/processed visualizations/static visualizations/interactive dashboard/static dashboard/templates

# 安裝依賴
pip install -r requirements.txt

# 執行資料收集和處理
python scripts/01_data_collection.py
python scripts/02_preprocessing.py

# 執行分析
python scripts/03_topic_analysis.py
python scripts/04_speaker_network.py
python scripts/05_conference_stats.py

# 啟動儀表板
python app.py

echo "專案設置完成並執行了所有腳本!"
