import sqlite3
import streamlit as st
import nltk
from nltk.corpus import wordnet as wn
import os

# アプリケーション内でのDBファイル名
DB_FILE = "wnjpn.db"

# --- 【初回来起動時のみ】NLTKデータのロード ---
@st.cache_resource
def load_nltk():
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('omw-2.0', quiet=True)

load_nltk()

# --- 【初回来起動時のみ】SQLite 接続用関数 ---
def get_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

# ==============================================================================
# 🛠️ データ取得用関数（SQLite版）
# ==============================================================================
@st.cache_data
def get_synsets_by_word_db(word):
    conn = get_connection()
    cur = conn.cursor()
    query = """
        SELECT DISTINCT s.synset, s.pos, s.name 
        FROM word w 
        JOIN sense se ON w.wordid = se.wordid 
        JOIN synset s ON se.synset = s.synset 
        WHERE w.lemma = ? AND w.lang IN ('jpn', 'eng')
    """
    cur.execute(query, (word,))
    rows = cur.fetchall()
    return [{"id": r[0], "pos": r[1], "name": r[2]} for r in rows]

@st.cache_data
def get_definitions_all_langs_db(synset_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT lang, def FROM synset_def WHERE synset = ?", (synset_id,))
    rows = cur.fetchall()
    
    defs = {"jpn": "なし", "eng": "なし"}
    extra_examples = []
    
    for lang, definition in rows:
        if lang in defs:
            parts = [p.strip() for p in definition.split(";")]
            if parts:
                defs[lang] = parts[0]
                if len(parts) > 1:
                    extra_examples.extend([p for p in parts[1:] if p])
                    
    return defs["jpn"], defs["eng"], extra_examples

@st.cache_data
def get_examples_db(synset_id, lang="eng"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT def FROM synset_ex WHERE synset = ? AND lang = ?", (synset_id, lang))
    return [r[0] for r in cur.fetchall()]

@st.cache_data
def get_lemmas_db(synset_id, lang):
    conn = get_connection()
    cur = conn.cursor()
    query = """
        SELECT w.lemma FROM sense se 
        JOIN word w ON se.wordid = w.wordid 
        WHERE se.synset = ? AND w.lang = ?
    """
    cur.execute(query, (synset_id, lang))
    return [r[0] for r in cur.fetchall()]

@st.cache_data
def get_related_synsets_db(synset_id, link_type):
    conn = get_connection()
    cur = conn.cursor()
    query = """
        SELECT s.synset, s.pos, s.name FROM synlink sl 
        JOIN synset s ON sl.synset2 = s.synset 
        WHERE sl.synset1 = ? AND sl.link = ?
    """
    cur.execute(query, (synset_id, link_type))
    rows = cur.fetchall()
    return [{"id": r[0], "pos": r[1], "name": r[2]} for r in rows]

# ==============================================================================
# UI 設定 & 多言語辞書 (Localization)
# ==============================================================================
st.set_page_config(page_title="Japanese WordNet Search", layout="wide", page_icon="🚀")

# 🌐 言語切り替えラジオボタンの配置
lang_choice = st.sidebar.radio("🌐 Language / 言語", ["日本語", "English"])
is_eng = (lang_choice == "English")

# 品詞マップのローカライズ
POS_MAP = {
    'n': 'Noun (名詞)' if is_eng else '名詞 (Noun)',
    'v': 'Verb (動詞)' if is_eng else '動詞 (Verb)',
    'a': 'Adjective (形容詞)' if is_eng else '形容詞 (Adjective)',
    's': 'Satellite Adjective (形容詞)' if is_eng else '形容詞 (Satellite Adjective)',
    'r': 'Adverb (副詞)' if is_eng else '副詞 (Adverb)'
}

# 画面全体の文言を一括管理する辞書
UI = {
    "about_title": "🚀 About App" if is_eng else "🚀 アプリについて",
    "about_desc": (
        "**Japanese WordNet Search** is a tool for exploring Japanese and English data from the 'WordNet' concept dictionary. "
        "Enter a word to concurrently retrieve Synsets, synonyms, definitions, hypernyms/hyponyms, and corresponding bilingual terms."
        if is_eng else
        "**Japanese WordNet Search** は、概念辞書「WordNet」の日本語・英語データを検索できるツールです。"
        "単語を入力すると**Synset（類義関係のセット）・類義語・概念定義・上位語や下位語、および対応する日英の類語**をまとめて検索できます。"
    ),
    "how_to": "📖 How to Use" if is_eng else "📖 使い方",
    "how_to_desc": (
        "1. Activate the Japanese WordNet DB from the sidebar if needed (first time only).\n"
        "2. Type a word into the search box (supports both Japanese and English).\n"
        "   * *Note: Only exact matches will be returned.*"
        if is_eng else
        "1. 最初のみ、必要に応じて日本語WordNet DBをロードしてください。\n"
        "2. メイン画面の検索ボックスに、調べたい単語（日本語または英語）を入力します。\n"
        "   * ※ 完全一致した単語のみヒットします。"
    ),
    "db_unloaded": "⚠️ Japanese WordNet DB (wnjpn.db) is not loaded." if is_eng else "⚠️ 日本語専用DB（wnjpn.db）が未ロードです。",
    "db_btn": "📦 Activate Database" if is_eng else "📦 データベースを有効化する",
    "db_loading": "Loading data into cache... (Takes dozens of seconds)" if is_eng else "データをキャッシュに読み込み中...（数十秒かかります）",
    "db_success": "Database loaded successfully!" if is_eng else "データベースのロードに成功しました！",
    "db_error": "Error occurred during loading:" if is_eng else "ロード中にエラーが発生しました:",
    "db_loaded": "✅ Japanese WordNet DB: Loaded (Cache Active)" if is_eng else "✅ 日本語専用DB: ロード済み (キャッシュ有効)",
    
    "src_diff": "📊 Data Source Differences" if is_eng else "📊 データソースの違い",
    "src_desc": (
        "Results are retrieved from two data sources simultaneously:\n\n"
        "* **Japanese WordNet DB**\n"
        "  * Based on Princeton WordNet, containing specific Japanese expressions and unique mappings.\n"
        "* **Open Multilingual Wordnet (OMW) DB**\n"
        "  * Integrated framework provided via NLTK, facilitating standardized multi-source cross-lingual access."
        if is_eng else
        "検索時、以下2つのデータソースから同時に結果を取得します。\n\n"
        "* **日本語WordNet DB**\n"
        "  * **特徴:** 日本語WordNet公式のSQLite版。プリンストン大学の英語WordNetをベースにしつつ、日本語特有の表現や概念（synset）を含んでいる。\n"
        "* **Open Multilingual Wordnet (OMW) DB**\n"
        "  * **特徴:** NLTKのOpen Multilingual Wordnet (OMW) ライブラリを利用。日本語を含む世界各国の多様な言語のWordNetを共通のIDや形式で一括して検索・利用できるように統合した枠組み。"
    ),
    "what_is_wn": "💡 What is WordNet?" if is_eng else "💡 WordNetとは？",
    "wn_desc": (
        "WordNet is a lexical database that groups words into sets of synonyms called 'Synsets', structuring them into conceptual hierarchies."
        if is_eng else
        "WordNetは、単語を「概念（Synset：類義語の集合）という単位でまとめた辞書です。各Synsetは他のsynsetと意味的に結びついており、上位下位の階層構造でできています。"
    ),
    
    "main_title": "🚀 Japanese WordNet Search",
    "main_desc": (
        "Enter a word to discover its Synsets, synonyms, hypernyms (parent concepts), and hyponyms (child concepts)."
        if is_eng else
        "単語を入力して検索すると、WordNetでのSynset（概念）と類義語、各Synsetの上位語（親概念）や下位語（子概念）が表示されます。"
    ),
    "input_label": "Enter a word (Supports Japanese/English):" if is_eng else "検索したい単語を入力してください（日本語・英語両対応）：",
    "placeholder": "e.g., study, run, 勉強" if is_eng else "例: 勉強、study、走る、run",
    
    "summary_title": "📋 Search Hit Summary" if is_eng else "📋 検索ヒット件数",
    "lbl_db": "Japan WordNet" if is_eng else "🗾日本語WordNet",
    "lbl_omw": "OMW (NLTK)" if is_eng else "🌍OMW",
    "db_not_ready": "Not Loaded" if is_eng else "未ロード",
    "db_not_ready_delta": "Can be activated from sidebar" if is_eng else "サイドバーから有効化できます",
    
    "tab_title_db": "🗾 Japanese WordNet Results" if is_eng else "🗾日本語WordNetの結果",
    "tab_title_omw": "🌍 OMW Results" if is_eng else "🌍OMWの結果",
    "no_results": "No matching concepts found for" if is_eng else "一致する概念は見つかりませんでした：",
    
    "lbl_def": "**📝 Definitions:**" if is_eng else " **📝 概念の定義 (Definitions):**",
    "lbl_def_ja": "- 🇯🇵 **Japanese:** " if is_eng else "- 🇯🇵 **日本語:** ",
    "lbl_def_en": "- 🇺🇸 **English:** " if is_eng else "- 🇺🇸 **英語:** ",
    "lbl_ex": "**📖 Examples:**" if is_eng else "**📖 例文 (Examples):**",
    "lbl_no_ex": "No examples available" if is_eng else "例文なし",
    "lbl_syn_ja": "**🇯🇵 Japanese Synonyms:**" if is_eng else "**🇯🇵 日本語の類義語:**",
    "lbl_syn_en": "**🇺🇸 English Synonyms:**" if is_eng else "**🇺🇸 英語の類義語 (Synonyms):**",
    "lbl_none": "None" if is_eng else "なし",
    
    "lbl_rel": "**🌿 Hierarchies & Sister Terms:**" if is_eng else "**🌿 概念の親子・兄弟関係 (Hierarchies & Sister Terms):**",
    "rel_hyper": "🔺 Hypernyms / Parents" if is_eng else "🔺 上位語 / 親概念",
    "rel_sib": "🔹 Sister Terms / Siblings" if is_eng else "🔹 兄弟語 / 同階層",
    "rel_hypo": "🔻 Hyponyms / Children" if is_eng else "🔻 下位語 / 子概念",
    "no_hyper": "Top-level concept." if is_eng else "最上位の概念です。",
    "no_sib": "No sister concepts." if is_eng else "兄弟概念なし。",
    "no_hypo": "Bottom-level concept." if is_eng else "最下位の概念です。",
    "omw_def_note": "- 🇺🇸 **English:** " if is_eng else "- 🇺🇸 **英語:** ",
    "init_info": "💡 Please enter a word in the search box above and press Enter." if is_eng else "💡上の検索ボックスに調べたい単語を入力して Enter を押してください。"
}

# --- データベースの存在チェック ---
db_exists = os.path.exists(DB_FILE)

# ==============================================================================
# 🗂️ サイドバー（アプリ説明・案内）
# ==============================================================================
st.sidebar.title(UI["about_title"])
st.sidebar.markdown(UI["about_desc"])

st.sidebar.markdown("---")
st.sidebar.subheader(UI["how_to"])
st.sidebar.markdown(UI["how_to_desc"])

# --- ロード処理の配置 ---
if not db_exists:
    st.sidebar.warning(UI["db_unloaded"])
    if st.sidebar.button(UI["db_btn"]):
        import urllib.request
        import gzip
        import shutil
        
        with st.sidebar.spinner(UI["db_loading"]):
            try:
                url = "https://github.com/bond-lab/wnja/releases/download/v1.1/wnjpn.db.gz"
                gz_file = "wnjpn.db.gz"
                urllib.request.urlretrieve(url, gz_file)
                with gzip.open(gz_file, 'rb') as f_in:
                    with open(DB_FILE, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                if os.path.exists(gz_file):
                    os.remove(gz_file)
                
                st.sidebar.success(UI["db_success"])
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"{UI['db_error']} {e}")
else:
    st.sidebar.success(UI["db_loaded"])

st.sidebar.markdown("---")
st.sidebar.subheader(UI["src_diff"])
st.sidebar.markdown(UI["src_desc"])

st.sidebar.markdown("---")
st.sidebar.subheader(UI["what_is_wn"])
st.sidebar.markdown(UI["wn_desc"])
st.sidebar.markdown("""
🔗 [WordNet](https://wordnet.princeton.edu/)  
🔗 [Japanese WordNet (NICT)](https://bond-lab.github.io/wnja/)  
🔗 [Open Multilingual Wordnet](https://omwn.org/)
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "日本語ワードネット （1.1版）© 2009-2011 NICT, 2012-2015 Francis Bond and 2016-2024 Francis Bond, Takayuki Kuribayashi\n\n"
    "Linked to [Japanese Wordnet](https://bond-lab.github.io/wnja/index.ja.html)"
)
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Japanese WordNet Search UI. Powered by Streamlit & NLTK.")

# ==============================================================================
# 💻 メインボディ（メイン画面）
# ==============================================================================
st.title(UI["main_title"])
st.markdown(UI["main_desc"])

# 🔍 検索キーワード入力
word_input = st.text_input(UI["input_label"], value="", placeholder=UI["placeholder"])

if word_input:
    # 1. NLTK版の取得
    all_synsets_nltk = wn.synsets(word_input, lang='jpn') + wn.synsets(word_input, lang='eng')
    seen_synsets = set()
    unique_synsets_nltk = []
    for syn in all_synsets_nltk:
        if syn.name() not in seen_synsets:
            seen_synsets.add(syn.name())
            unique_synsets_nltk.append(syn)
    count_nltk = len(unique_synsets_nltk)
    
    # 2. ローカルDB版の取得
    if db_exists:
        all_synsets_db = get_synsets_by_word_db(word_input)
        count_db = len(all_synsets_db)
    else:
        all_synsets_db = []
        count_db = 0

    # 📊 取得件数のサマリー表示
    st.markdown(f"### {UI['summary_title']}")
    col_count1, col_count2 = st.columns(2)
    with col_count1:
        if db_exists:
            st.metric(label=UI["lbl_db"], value=f"{count_db} hits")
        else:
            st.metric(label=UI["lbl_db"], value=UI["db_not_ready"], delta=UI["db_not_ready_delta"], delta_color="off")
    with col_count2:
        st.metric(label=UI["lbl_omw"], value=f"{count_nltk} hits")

    st.markdown("---")

    if db_exists:
        tab_db, tab_nltk = st.tabs([UI["tab_title_db"], UI["tab_title_omw"]])
    else:
        tab_nltk = st.tabs([UI["tab_title_omw"]])[0]
        tab_db = None

    # --------------------------------------------------------------------------
    # 【タブ1】日本語専用DBの結果
    # --------------------------------------------------------------------------
    if tab_db and db_exists:
        with tab_db:
            if not all_synsets_db:
                st.info(f"{UI['no_results']} 『{word_input}』")
            else:
                pos_groups_db = {}
                for syn in all_synsets_db:
                    pos = syn['pos']
                    if pos not in pos_groups_db: pos_groups_db[pos] = []
                    pos_groups_db[pos].append(syn)
                
                sub_tabs_db = st.tabs([f"{POS_MAP.get(pos, pos)} ({len(synsets)}hits)" for pos, synsets in pos_groups_db.items()])
                for sub_tab, (pos, synsets) in zip(sub_tabs_db, pos_groups_db.items()):
                    with sub_tab:
                        for syn in synsets:
                            syn_id = syn['id']
                            syn_name = syn['name']
                            with st.container(border=True):
                                st.markdown(f"#### 💡 Synset ID: `{syn_name}`")
                                
                                # 1. 定義 (Definitions)
                                def_ja, def_en, extra_exs = get_definitions_all_langs_db(syn_id)
                                st.markdown(UI["lbl_def"])
                                st.markdown(f"{UI['lbl_def_ja']}{def_ja}")
                                st.markdown(f"{UI['lbl_def_en']}*{def_en}*")
                                
                                # 2. 例文 (Examples)
                                examples_en = get_examples_db(syn_id, lang="eng")
                                examples_ja = get_examples_db(syn_id, lang="jpn")
                                all_examples = list(set(examples_en + examples_ja + extra_exs))
                                
                                st.markdown(UI["lbl_ex"])
                                if all_examples:
                                    for ex in all_examples: 
                                        st.markdown(f"- *{ex}*")
                                else:
                                    st.caption(UI["lbl_no_ex"])
                                
                                # 3. 類義語 (Synonyms)
                                ja_lemmas = get_lemmas_db(syn_id, lang="jpn")
                                en_lemmas = get_lemmas_db(syn_id, lang="eng")
                                
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown(UI["lbl_syn_ja"])
                                    st.info(", ".join(ja_lemmas) if ja_lemmas else UI["lbl_none"])
                                with c2:
                                    st.markdown(UI["lbl_syn_en"])
                                    st.success(", ".join(en_lemmas) if en_lemmas else UI["lbl_none"])
                                    
                                # 4. 概念の親子・兄弟関係
                                st.markdown(UI["lbl_rel"])
                                rel_c1, rel_c2, rel_c3 = st.columns(3)
                                
                                hypernyms = get_related_synsets_db(syn_id, "hype")
                                with rel_c1:
                                    with st.expander(f"{UI['rel_hyper']} ({len(hypernyms)}hits)"):
                                        if hypernyms:
                                            for hyper in hypernyms:
                                                h_ja = get_lemmas_db(hyper['id'], lang="jpn")
                                                st.markdown(f"• **`{hyper['name']}`**")
                                                st.caption(f"└ 日: {', '.join(h_ja[:5])}")
                                        else: st.write(UI["no_hyper"])
                                        
                                with rel_c2:
                                    siblings = []
                                    sibling_ids = set()
                                    for hyper in hypernyms:
                                        for sib in get_related_synsets_db(hyper['id'], "hypo"):
                                            if sib['id'] != syn_id and sib['id'] not in sibling_ids:
                                                siblings.append(sib)
                                                sibling_ids.add(sib['id'])
                                    with st.expander(f"{UI['rel_sib']} ({len(siblings)}hits)"):
                                        if siblings:
                                            for sib in siblings:
                                                sib_ja = get_lemmas_db(sib['id'], lang="jpn")
                                                st.markdown(f"• **`{sib['name']}`**")
                                                st.caption(f"└ 日: {', '.join(sib_ja[:5])}")
                                        else: st.write(UI["no_sib"])
                                        
                                with rel_c3:
                                    hyponyms = get_related_synsets_db(syn_id, "hypo")
                                    with st.expander(f"{UI['rel_hypo']} ({len(hyponyms)}hits)"):
                                        if hyponyms:
                                            for hypo in hyponyms:
                                                hypo_ja = get_lemmas_db(hypo['id'], lang="jpn")
                                                st.markdown(f"• **`{hypo['name']}`**")
                                                st.caption(f"└ 日: {', '.join(hypo_ja[:5])}")
                                        else: st.write(UI["no_hypo"])

    # --------------------------------------------------------------------------
    # 【タブ2】Multilingual (NLTK) の結果
    # --------------------------------------------------------------------------
    with tab_nltk:
        if not unique_synsets_nltk:
            st.info(f"{UI['no_results']} 『{word_input}』")
        else:
            pos_groups_nltk = {}
            for syn in unique_synsets_nltk:
                pos = syn.pos()
                if pos not in pos_groups_nltk: pos_groups_nltk[pos] = []
                pos_groups_nltk[pos].append(syn)
            
            sub_tabs_nltk = st.tabs([f"{POS_MAP.get(pos, pos)} ({len(synsets)}hits)" for pos, synsets in pos_groups_nltk.items()])
            for sub_tab, (pos, synsets) in zip(sub_tabs_nltk, pos_groups_nltk.items()):
                with sub_tab:
                    for syn in synsets:
                        with st.container(border=True):
                            st.markdown(f"#### 💡 Synset ID: `{syn.name()}`")
                            
                            # 1. 定義 (Definitions)
                            st.markdown(UI["lbl_def"])
                            st.markdown(f"{UI['omw_def_note']}*{syn.definition()}*")
                            
                            # 2. 例文 (Examples)
                            examples = syn.examples()
                            st.markdown(UI["lbl_ex"])
                            if examples:
                                for ex in examples: st.markdown(f"- *\"{ex}\"*")
                            else:
                                st.caption(UI["lbl_no_ex"])
                            
                            # 3. 類義語 (Synonyms)
                            ja_lemmas = syn.lemma_names('jpn')
                            en_lemmas = syn.lemma_names('eng')
                            
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown(UI["lbl_syn_ja"])
                                st.info(", ".join(ja_lemmas) if ja_lemmas else UI["lbl_none"])
                            with c2:
                                st.markdown(UI["lbl_syn_en"])
                                st.success(", ".join(en_lemmas) if en_lemmas else UI["lbl_none"])
                                
                            # 4. 概念の親子・兄弟関係
                            st.markdown(UI["lbl_rel"])
                            rel_c1, rel_c2, rel_c3 = st.columns(3)

                            hypernyms = syn.hypernyms()
                            with rel_c1:
                                with st.expander(f"{UI['rel_hyper']} ({len(hypernyms)}hits)"):
                                    if hypernyms:
                                        for hyper in hypernyms:
                                            st.markdown(f"• **`{hyper.name()}`**")
                                            st.caption(f"└ 日: {', '.join(hyper.lemma_names('jpn')[:5])}")
                                    else: st.write(UI["no_hyper"])                                    
                            with rel_c2:
                                siblings = []
                                for hyper in hypernyms:
                                    for sibling in hyper.hyponyms():
                                        if sibling != syn and sibling not in siblings: siblings.append(sibling)
                                with st.expander(f"{UI['rel_sib']} ({len(siblings)}hits)"):
                                    if siblings:
                                        for sib in siblings:
                                            st.markdown(f"• **`{sib.name()}`**")
                                            st.caption(f"└ 日: {', '.join(sib.lemma_names('jpn')[:5])}")
                                    else: st.write(UI["no_sib"])
                            with rel_c3:
                                hyponyms = syn.hyponyms()
                                with st.expander(f"{UI['rel_hypo']} ({len(hyponyms)}hits)"):
                                    if hyponyms:
                                        for hypo in hyponyms:
                                            st.markdown(f"• **`{hypo.name()}`**")
                                            st.caption(f"└ 日: {', '.join(hypo.lemma_names('jpn')[:5])}")
                                    else: st.write(UI["no_hypo"])
else:
    st.info(UI["init_info"])