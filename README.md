# Japanese WordNet Search

日本語WordNetのデータを視覚的かつ直感的に検索・探索するためのStreamlit Webアプリケーション。  
A Streamlit web application designed to visually and intuitively search and explore Japanese WordNet data.

👉 [Live Demo / Webアプリで試す](https://ja-wordnet-search.streamlit.app/)

---

## 主な機能 / Key Features

### 1. 2つのデータソースからの一括検索 / Multi-Source Search
* **日本語WordNet DB (SQLite版):** 公式のSQLiteデータベース（`wnjpn.db`）から直接データを検索します。（※初回起動時にサイドバーからダウンロード可能）  
  **SQLite Version:** Fast, direct search from the official Japanese WordNet SQLite database (`wnjpn.db`). (Can be downloaded from the sidebar on first run).
* **Open Multilingual Wordnet (OMW):** NLTK経由で多言語WordNetの日本語・英語データを並行して取得し、結果を比較できます。  
  **OMW Version:** Simultaneously retrieves and compares data using NLTK's Open Multilingual Wordnet (OMW) for both Japanese and English.

### 2. 品詞別に整理 / Organized by POS
* 検索結果は自動的に「名詞」「動詞」「形容詞」「副詞」などの品詞（Part of Speech）ごとに分類されます。  
  Automatically categorizes search results by Part of Speech (POS) such as Noun, Verb, Adjective, and Adverb using a clean tabbed interface.


### 3. 概念の構造を表示 / Retrieve Concept Networks
* 選択した概念（Synset）の「上位語（親概念）」「下位語（子概念）」「兄弟語（同階層）」も表示されます。  
 Extracts and displays "Hypernyms (Parents)", "Hyponyms (Children)", and "Sister Terms (Siblings)" for any selected Synset
* 英語の定義文や例文に加え、日本語と英語それぞれの類義語リストを並べて表示します。  
   Displays English definitions and examples alongside side-by-side synonym lists for both Japanese and English.

---

## 必要要件 / Requirements

* Python 3.8+
* streamlit
* nltk

---

## 使い方 / Quick Start

### 1. 依存ライブラリのインストール / Install Dependencies
```bash
pip install streamlit nltk
```

### 2. アプリケーションの起動 / Run the App
```bash
streamlit run app.py
```

### 3. データの準備 / Data Preparation
- メイン画面の検索ボックスに、調べたい単語（日本語または英語）を入力して検索します。  
  Enter the word you want to search (supports both Japanese and English) in the search box on the main screen

- 日本語WordNet DB（wnjpn.db）を利用する場合は、サイドバーにある「📦 データベースを有効化する」ボタンをクリックすると、データが自動的に読み込まれ、即座に検索可能になります。  
 To use the Japanese WordNet DB (wnjpn.db), click the "📦 データベースを有効化する (Activate Database)" button in the sidebar. The data will be automatically loaded into the cache and become searchable immediately.


## ライセンスとクレジット / License & Credits
本アプリケーションは、以下のオープンデータおよびコーパスを利用しています。  
This application utilizes the following open data and corpora

### 日本語 WordNet (Japanese Wordnet)
- 日本語ワードネット （1.1版）/ apanese Wordnet (v1.1) © 2009-2011 NICT, 2012-2015 Francis Bond and 2016-2024 Francis Bond, Takayuki Kuribayashi

- 配布元 / Source : [Japanese Wordnet Official Site](https://bond-lab.github.io/wnja/index.ja.html)

- ライセンス / Licence: 詳細については、本リポジトリ内の LICENSE_wnja.txt をご確認ください。  
  For details about the license, please check LICENSE_wnja.txt in this repository.

### Princeton WordNet
- WordNet Release 3.0 Copyright 2006 by Princeton University. All rights reserved.

- 配布元 / Source : [Princeton WordNet](https://wordnet.princeton.edu/)

### Open Multilingual Wordnet (OMW)
- 配布元 / Source : [Open Multilingual Wordnet](https://omwn.org/)
