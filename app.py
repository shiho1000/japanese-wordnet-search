import sqlite3
import streamlit as st
import nltk
from nltk.corpus import wordnet as wn
import os
import urllib.request
import gzip
import shutil

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
        WHERE w.lemma = ? AND w.lang = 'jpn'
    """
    cur.execute(query, (word,))
    rows = cur.fetchall()
    return [{"id": r[0], "pos": r[1], "name": r[2]} for r in rows]

@st.cache_data
def get_definition_db(synset_id, lang="eng"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT def FROM synset_def WHERE synset = ? AND lang = ?", (synset_id, lang))
    row = cur.fetchone()
    return row[0] if row else "なし"

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
# UI 設定
# ==============================================================================
POS_MAP = {
    'n': '名詞 (Noun)',
    'v': '動詞 (Verb)',
    'a': '形容詞 (Adjective)',
    's': '形容詞 (Satellite Adjective)',
    'r': '副詞 (Adverb)'
}

st.set_page_config(page_title="Japanese WordNet Search", layout="wide", page_icon="🚀")

# 🎨 親タブ（1階層目）をボタン・ボックス型にするためのカスタムCSS
st.markdown("""
    <style>
    /* 1階層目の大枠のタブリストのみに適用するデザイン設定 */
    div[data-testid="stTabBlock"] > div > [data-testid="stHorizontalBlock"] div[role="tablist"] {
        gap: 10px;
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    /* 未選択状態の親タブボタン */
    div[data-testid="stTabBlock"] > div > [data-testid="stHorizontalBlock"] button[role="tab"] {
        background-color: #ffffff;
        border: 1px solid #dcdcdc !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        height: auto !important;
        font-weight: bold;
        color: #555555;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
    }
    /* ホバーしたとき */
    div[data-testid="stTabBlock"] > div > [data-testid="stHorizontalBlock"] button[role="tab"]:hover {
        background-color: #f1f3f5;
        color: #000000;
    }
    /* 選択状態の親タブボタン（赤線を消して背景色と枠線でアピール） */
    div[data-testid="stTabBlock"] > div > [data-testid="stHorizontalBlock"] button[role="tab"][aria-selected="true"] {
        background-color: #ff4b4b !important;
        color: #ffffff !important;
        border-color: #ff4b4b !important;
    }
    /* 選択中タブの下の赤いインジケーター線を完全に消去 */
    div[data-testid="stTabBlock"] > div > [data-testid="stHorizontalBlock"] button[role="tab"] div[data-testid="stMarkdownContainer"] + div {
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- データベースの存在チェック ---
db_exists = os.path.exists(DB_FILE)

# ==============================================================================
# 🗂️ サイドバー（アプリ説明・案内）
# ==============================================================================
st.sidebar.title("🚀About App")
st.sidebar.markdown("""
**Japanese WordNet Search** は、概念辞書「WordNet」の日本語データを検索できるツールです。
日本語の単語を入力すると**Synset（類義関係のセット）・類義語・概念定義・上位語（親概念）や下位語（子概念）、および対応する英語の類語**をまとめて検索できます。
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📖 使い方")
st.sidebar.markdown("""
1. 最初のみ、必要に応じて日本語WordNet DBをダウンロードしてください（ダウンロードしなくても利用できます。その場合は、Open Multilingual Wordnet (OMW)からのみ検索します）。
2. メイン画面の検索ボックスに、調べたい日本語の単語を入力します。
   * ※ 漢字・ひらがな・カタカナは区別されます。
   * ※ 完全一致した単語のみヒットします。
""")

# --- ダウンロード処理の配置 ---
if not db_exists:
    st.sidebar.warning("⚠️ 日本語WordNetDBが未セットアップです。")
    if st.sidebar.button("📦 日本語WordNet DBをDLする (約200MB)"):
        with st.spinner("DL中...（数分かかる場合があります）"):
            url = "https://github.com/bond-lab/wnja/releases/download/v1.1/wnjpn.db.gz"
            gz_file = "wnjpn.db.gz"
            urllib.request.urlretrieve(url, gz_file)
            with gzip.open(gz_file, 'rb') as f_in:
                with open(DB_FILE, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            if os.path.exists(gz_file):
                os.remove(gz_file)
        st.sidebar.success("DL完了！再読み込みします。")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📊 データソースの違い")
st.sidebar.markdown("""
検索時、以下2つのデータソースから同時に結果を取得します。

* **日本語WordNet DB**
  * **特徴:** 日本語WordNet公式のSQLite版。プリンストン大学の英語WordNetをベースにしつつ、日本語特有の表現や概念（synset）を含んでいる。
* **Open Multilingual Wordnet (OMW) DB**
  * **特徴:** NLTKのOpen Multilingual Wordnet (OMW) ライブラリを利用。日本語を含む世界各国の多様な言語のWordNetを共通のIDや形式で一括して検索・利用できるように統合した枠組み。このアプリでは英語と日本語のみ表示可能です。

両者は、語彙のカバー範囲やマッピングが異なっていることもあります。
""")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 WordNetとは？")
st.sidebar.markdown("""
WordNetは、単語を「概念（Synset：類義語の集合）という単位でまとめた辞書です。各Synsetは他のsynsetと意味的に結びついており、上位下位の階層構造でできています。

🔗 [WordNetをもっと詳しく](https://wordnet.princeton.edu/)  
🔗 [日本語WordNet（NICT）](https://bond-lab.github.io/wnja/)  
🔗 [Open Multilingual Wordnet](https://omwn.org/)
""")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Japanese WordNet Search UI. Powered by Streamlit & NLTK.")


# ==============================================================================
# 💻 メインボディ（メイン画面）
# ==============================================================================
st.title("🚀 Japanese WordNet Search")
st.markdown("日本語の単語を入力して検索すると、WordNetでのSynset（概念）と類義語、各Synsetの階層構造を表示します。")

# 🔍 検索キーワード入力
word_input = st.text_input("検索したい単語を入力してください：", value="", placeholder="例: 勉強、本、走る")

# 検索ワードが入力された場合のみ処理を実行
if word_input:
    # 双方からデータ件数を事前取得
    # 1. NLTK版の取得
    all_synsets_nltk = wn.synsets(word_input, lang='jpn')
    count_nltk = len(all_synsets_nltk)
    
    # 2. ローカルDB版の取得（存在する場合のみ）
    if db_exists:
        all_synsets_db = get_synsets_by_word_db(word_input)
        count_db = len(all_synsets_db)
    else:
        all_synsets_db = []
        count_db = 0

    # 📊 取得件数のサマリー表示
    st.markdown("### 📋 検索ヒット件数")
    col_count1, col_count2 = st.columns(2)
    with col_count1:
        if db_exists:
            st.metric(label="🗾日本語WordNet", value=f"{count_db} 件")
        else:
            st.metric(label="🗾日本語WordNet", value="未DL", delta="サイドバーから有効化できます", delta_color="off")
    with col_count2:
        # ※ 構文エラーになっていた箇所の修正（ダブルクォーテーションの閉じ忘れ補正）
        st.metric(label="🌍OMW", value=f"{count_nltk} 件")

    st.markdown("---")

    # ➔ 結果はタブで切り替え
    # DBが利用可能なら2つのタブ、未DLならNLTKタブのみ表示
    if db_exists:
        tab_db, tab_nltk = st.tabs(["🗾日本語WordNetの結果", "🌍OMWの結果"])
    else:
        tab_nltk = st.tabs(["🌍OMWの結果"])[0]
        tab_db = None

    # --------------------------------------------------------------------------
    # 【タブ1】日本語専用DBの結果
    # --------------------------------------------------------------------------
    if tab_db and db_exists:
        with tab_db:
            if not all_synsets_db:
                st.info(f"日本語WordNet：『{word_input}』に一致する概念は見つかりませんでした。")
            else:
                pos_groups_db = {}
                for syn in all_synsets_db:
                    pos = syn['pos']
                    if pos not in pos_groups_db: pos_groups_db[pos] = []
                    pos_groups_db[pos].append(syn)
                
                # 品詞ごとのサブタブ
                sub_tabs_db = st.tabs([POS_MAP.get(pos, pos) for pos in pos_groups_db.keys()])
                for sub_tab, (pos, synsets) in zip(sub_tabs_db, pos_groups_db.items()):
                    with sub_tab:
                        for syn in synsets:
                            syn_id = syn['id']
                            syn_name = syn['name']
                            with st.container(border=True):
                                st.markdown(f"#### 💡 Synset ID: `{syn_name}`")
                                definition = get_definition_db(syn_id, lang="eng")
                                st.markdown(f"**📝 英語の定義 (Definition):**  \n*{definition}*")
                                
                                examples = get_examples_db(syn_id, lang="eng")
                                if examples:
                                    st.markdown("**📖 例文 (Examples):**")
                                    for ex in examples: st.markdown(f"- *\"{ex}\"*")
                                
                                ja_lemmas = get_lemmas_db(syn_id, lang="jpn")
                                en_lemmas = get_lemmas_db(syn_id, lang="eng")
                                
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown("**🇯🇵 日本語の類義語:**")
                                    st.info(", ".join(ja_lemmas) if ja_lemmas else "なし")
                                with c2:
                                    st.markdown("**🇺🇸 英語の類義語 (Synonyms):**")
                                    st.success(", ".join(en_lemmas) if en_lemmas else "なし")
                                    
                                st.markdown("**🌿 概念の親子・兄弟関係 (Hierarchies & Sister Terms):**")
                                rel_c1, rel_c2, rel_c3 = st.columns(3)
                                
                                hypernyms = get_related_synsets_db(syn_id, "hype")
                                with rel_c1:
                                    with st.expander(f"🔺 上位語 / 親概念 ({len(hypernyms)}件)"):
                                        if hypernyms:
                                            for hyper in hypernyms:
                                                h_ja = get_lemmas_db(hyper['id'], lang="jpn")
                                                st.markdown(f"• **`{hyper['name']}`**")
                                                st.caption(f"└ 日: {', '.join(h_ja[:5])}")
                                        else: st.write("最上位の概念です。")
                                        
                                with rel_c2:
                                    siblings = []
                                    sibling_ids = set()
                                    for hyper in hypernyms:
                                        for sib in get_related_synsets_db(hyper['id'], "hypo"):
                                            if sib['id'] != syn_id and sib['id'] not in sibling_ids:
                                                siblings.append(sib)
                                                sibling_ids.add(sib['id'])
                                    with st.expander(f"🔹 兄弟語 / 同階層 ({len(siblings)}件)"):
                                        if siblings:
                                            for sib in siblings:
                                                sib_ja = get_lemmas_db(sib['id'], lang="jpn")
                                                st.markdown(f"• **`{sib['name']}`**")
                                                st.caption(f"└ 日: {', '.join(sib_ja[:5])}")
                                        else: st.write("兄弟概念なし。")
                                        
                                with rel_c3:
                                    hyponyms = get_related_synsets_db(syn_id, "hypo")
                                    with st.expander(f"🔻 下位語 / 子概念 ({len(hyponyms)}件)"):
                                        if hyponyms:
                                            for hypo in hyponyms:
                                                hypo_ja = get_lemmas_db(hypo['id'], lang="jpn")
                                                st.markdown(f"• **`{hypo['name']}`**")
                                                st.caption(f"└ 日: {', '.join(hypo_ja[:5])}")
                                        else: st.write("最下位の概念です。")

    # --------------------------------------------------------------------------
    # 【タブ2】Multilingual (NLTK) の結果
    # --------------------------------------------------------------------------
    with tab_nltk:
        if not all_synsets_nltk:
            st.info(f"OMW：『{word_input}』に一致する概念は見つかりませんでした。")
        else:
            pos_groups_nltk = {}
            for syn in all_synsets_nltk:
                pos = syn.pos()
                if pos not in pos_groups_nltk: pos_groups_nltk[pos] = []
                pos_groups_nltk[pos].append(syn)
            
            # 品詞ごとのサブタブ
            sub_tabs_nltk = st.tabs([POS_MAP.get(pos, pos) for pos in pos_groups_nltk.keys()])
            for sub_tab, (pos, synsets) in zip(sub_tabs_nltk, pos_groups_nltk.items()):
                with sub_tab:
                    for syn in synsets:
                        with st.container(border=True):
                            st.markdown(f"#### 💡 Synset ID: `{syn.name()}`")
                            st.markdown(f"**📝 英語の定義 (Definition):**  \n*{syn.definition()}*")
                            
                            examples = syn.examples()
                            if examples:
                                st.markdown("**📖 例文 (Examples):**")
                                for ex in examples: st.markdown(f"- *\"{ex}\"*")
                            
                            ja_lemmas = syn.lemma_names('jpn')
                            en_lemmas = syn.lemma_names('eng')
                            
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**🇯🇵 日本語の類義語:**")
                                st.info(", ".join(ja_lemmas) if ja_lemmas else "なし")
                            with c2:
                                Princess = st.markdown("**🇺🇸 英語の類義語 (Synonyms):**")
                                st.success(", ".join(en_lemmas) if en_lemmas else "なし")
                                
                            st.markdown("**🌿 概念の親子・兄弟関係 (Hierarchies & Sister Terms):**")
                            rel_c1, rel_c2, rel_c3 = st.columns(3)

                            hypernyms = syn.hypernyms()
                            with rel_c1:
                                with st.expander(f"🔺 上位語 / 親概念 ({len(hypernyms)}件)"):
                                    if hypernyms:
                                        for hyper in hypernyms:
                                            st.markdown(f"• **`{hyper.name()}`**")
                                            st.caption(f"└ 日: {', '.join(hyper.lemma_names('jpn')[:5])}")
                                    else: st.write("最上位の概念です。")                                    
                            with rel_c2:
                                siblings = []
                                for hyper in hypernyms:
                                    for sibling in hyper.hyponyms():
                                        if sibling != syn and sibling not in siblings: siblings.append(sibling)
                                with st.expander(f"🔹 兄弟語 / 同階層 ({len(siblings)}件)"):
                                    if siblings:
                                        for sib in siblings:
                                            st.markdown(f"• **`{sib.name()}`**")
                                            st.caption(f"└ 日: {', '.join(sib.lemma_names('jpn')[:5])}")
                                    else: st.write("兄弟概念なし。")
                            with rel_c3:
                                hyponyms = syn.hyponyms()
                                with st.expander(f"🔻 下位語 / 子概念 ({len(hyponyms)}件)"):
                                    if hyponyms:
                                        for hypo in hyponyms:
                                            st.markdown(f"• **`{hypo.name()}`**")
                                            st.caption(f"└ 日: {', '.join(hypo.lemma_names('jpn')[:5])}")
                                    else: st.write("最下位の概念です。")
else:
    # 🔍 未入力時のプレースホルダー表示
    st.info("💡上の検索ボックスに調べたい単語（例：『辞書』『走る』など）を入力して Enter を押してください。")