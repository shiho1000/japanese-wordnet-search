# Japanese WordNet Search

日本語WordNetのデータを視覚的かつ直感的に検索・探索するためのStreamlit Webアプリケーション。  
A Streamlit web application designed to visually and intuitively search and explore Japanese WordNet data.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shiho1000-japanese-wordnet-search-app-ii1yqa.streamlit.app/)
👉 [Live Demo / Webアプリで試す](https://shiho1000-japanese-wordnet-search-app-ii1yqa.streamlit.app/)

---

## 主な機能 / Key Features

### 1. 2つのデータソースからの一括検索 / Multi-Source Search
* **日本語WordNet DB (SQLite版):** 公式のSQLiteデータベース（`wnjpn.db`）から直接データを検索します。（※初回起動時にサイドバーからダウンロード可能）
* **Open Multilingual Wordnet (OMW):** NLTK経由で多言語WordNetの日本語・英語データを並行して取得し、結果を比較できます。
* **SQLite Version:** Fast, direct search from the official Japanese WordNet SQLite database (`wnjpn.db`). (Can be downloaded from the sidebar on first run).
* **OMW Version:** Simultaneously retrieves and compares data using NLTK's Open Multilingual Wordnet (OMW) for both Japanese and English.

### 2. 品詞別に整理 / Organized by POS
* 検索結果は自動的に「名詞」「動詞」「形容詞」「副詞」などの品詞（Part of Speech）ごとに分類されます。
* Automatically categorizes search results by Part of Speech (POS) such as Noun, Verb, Adjective, and Adverb using a clean tabbed interface.


### 3. 概念の構造を表示 / Retrieve Concept Networks
* 選択した概念（Synset）の「上位語（親概念）」「下位語（子概念）」「兄弟語（同階層）」も表示されます
* 英語の定義文や例文に加え、日本語と英語それぞれの類義語リストを並べて表示します。
* Extracts and displays "Hypernyms (Parents)", "Hyponyms (Children)", and "Sister Terms (Siblings)" for any selected Synset (`st.expander`).
* Displays English definitions and examples alongside side-by-side synonym lists for both Japanese and English.

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