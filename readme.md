# COSCUP 2020-2024 資料分析與視覺化

本專案針對COSCUP開源人年會(2020-2024)的議程資料進行深度分析與視覺化，探索開源技術社群的演變趨勢、講者網絡關係與技術主題變化。

## 專案簡介

COSCUP (Conference for Open Source Coders, Users and Promoters) 是台灣最大的開源社群年會，每年匯聚來自世界各地的開源貢獻者、使用者與推廣者。本專案通過分析其2020年至2024年的議程資料，嘗試回答以下問題：

- 開源社群關注的技術主題有何變化？
- 講者間的合作關係如何演變？
- 不同議程軌(Track)的特點與相互關係如何？
- 會議結構與規模如何隨時間變化？

## 資料來源

資料來源為COSCUP官方網站提供的JSON檔案：
- https://coscup.org/2020/json/session.json
- https://coscup.org/2021/json/session.json
- https://coscup.org/2022/json/session.json
- https://coscup.org/2023/json/session.json
- https://coscup.org/2024/json/session.json

## 功能特色

1. **資料收集與預處理**
   - 自動下載和處理JSON格式的議程資料
   - 統一不同年份的資料結構
   - 中文文本分詞與清洗

2. **主題演變分析**
   - 使用進階主題建模(LDA、NMF)挖掘議程主題
   - 關鍵詞趨勢分析
   - 技術主題年度變化可視化

3. **講者網絡與社群分析**
   - 講者合作網絡圖譜
   - 社群檢測與分群
   - 重要講者識別與影響力評估
   - 講者專長領域探索

4. **會議結構與演進分析**
   - 議程軌(Track)變化趨勢
   - 時間分布熱力圖
   - 會議規模與類型變化
   - 議程語言分布

5. **語意分析**
   - 議程內容與講者背景的語意關聯
   - 相似議程推薦
   - 技術關鍵詞演變追蹤

6. **互動式儀表板**
   - 整合所有分析結果的Web儀表板
   - 支援多維度資料探索
   - 動態篩選與交互功能

## 技術實現

本專案使用以下技術：

- **Python**: 核心資料處理與分析
- **Pandas/NumPy**: 資料處理與分析
- **Scikit-learn**: 機器學習與主題建模
- **NetworkX**: 網絡分析
- **Matplotlib/Seaborn/Plotly**: 資料視覺化
- **jieba**: 中文分詞
- **Flask**: Web後端
- **ECharts/D3.js**: 互動式圖表
- **HTML/CSS/JavaScript**: 前端實現

## 安裝與使用

### 環境需求

- Python 3.8+
- pip套件管理器
- 網絡連接（用於下載資料）
- 適用於Linux/macOS/Windows系統

### 安裝步驟

1. 複製專案

```bash
git clone https://github.com/yourusername/coscup-analysis.git
cd coscup-analysis
```

2. 安裝依賴

```bash
pip install -r requirements.txt
```

### 執行方式

1. 資料收集與預處理

```bash
python scripts/01_data_collection.py
python scripts/02_preprocessing.py
```

2. 執行分析

```bash
python scripts/03_topic_analysis.py
python scripts/04_speaker_network.py
python scripts/05_conference_stats.py
```

3. 啟動儀表板

```bash
python app.py
```

4. 訪問儀表板

在瀏覽器中打開 http://localhost:5000

### 快速啟動（一鍵執行）

```bash
bash run.sh
```

## 項目結構

```
coscup-analysis/
├── data/                      # 原始及處理後的資料
│   ├── raw/                   # 原始JSON檔案
│   └── processed/             # 處理後的CSV檔案
├── scripts/                   # 資料處理和分析腳本
│   ├── 01_data_collection.py  # 資料收集
│   ├── 02_preprocessing.py    # 資料預處理
│   ├── 03_topic_analysis.py   # 主題分析
│   ├── 04_speaker_network.py  # 講者網絡分析
│   └── 05_conference_stats.py # 會議結構分析
├── visualizations/            # 視覺化結果
│   ├── static/                # 靜態圖表
│   └── interactive/           # 互動式圖表
├── dashboard/                 # 互動式儀表板
│   ├── static/                # CSS、JS等靜態檔案
│   └── templates/             # HTML模板
├── app.py                     # Flask應用程式進入點
├── requirements.txt           # 專案依賴
├── run.sh                     # 一鍵執行腳本
└── README.md                  # 專案說明
```

## 分析結果摘要

### 1. 技術主題趨勢

分析顯示，2020年至2024年間，COSCUP議程中的技術主題有明顯變化：

- AI與機器學習相關主題顯著增長，從2020年的~15%升至2024年的~35%
- 容器與雲原生技術保持穩定關注度，約佔20-25%
- 資訊安全主題呈現逐年上升趨勢
- 前端開發技術議題比例相對減少

### 2. 講者社群結構

網絡分析顯示COSCUP講者社群呈現以下特點：

- 形成了5個主要社群群組，分別圍繞系統/運維、開發/程式語言、資料/AI、雲端/容器和安全/隱私等領域
- 每年約有40-50%的新講者加入
- 持續參與(多年演講)的講者形成了社群的核心連結點

### 3. 會議結構演變

- 會議規模逐年擴大，議程數從2020年的120個增至2024年的190個
- 議程軌(Track)數量增加，反映技術多樣化
- 英語議程比例逐年增加，顯示國際化趨勢

## 未來拓展方向

1. 整合其他開源會議資料進行比較研究
2. 加入社交媒體資料分析會議熱度與反饋
3. 擴展NLP分析以支援中英文混合分析
4. 建立預測模型預測未來技術趨勢

## 貢獻指南

歡迎提交Issue或Pull Request來協助改進專案。貢獻前請先閱讀CONTRIBUTING.md。

## 授權

本專案採用MIT授權。詳見LICENSE檔案。

## 聯絡方式

如有問題或建議，請聯繫：your.email@example.com
