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
    # 日本語(jpn)と英語(eng)の両方の補題(lemma)から検索できるように拡張
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
    """DBから日本語と英語両方の定義を取得し、セミコロンで混入している例文を分離する"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT lang, def FROM synset_def WHERE synset = ?", (synset_id,))
    rows = cur.fetchall()
    
    defs = {"jpn": "なし", "eng": "なし"}
    extra_examples = []
    
    for lang, definition in rows:
        if lang in defs:
            # セミコロンで区切られた例文が混ざっている場合の分離処理
            parts = [p.strip() for p in definition.split(";")]
            if parts:
                defs[lang] = parts[0] # 最初の部分を純粋な定義とする
                if len(parts) > 1:
                    # 残りの部分（クォーテーションで囲まれていることが多い）を例文としてストック
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
# UI 設定
# ==============================================================================
POS_MAP = {
    'n': '名詞 (Noun)',
    'v': '動詞 (Verb)',
    'a': '形容詞 (Adjective)',
    's': '形容詞 (Satellite Adjective)',
    'r': '副詞 (Adverb)'
}

st.set_page_config(page_title="Japanese WordNet Search", layout="wide", page_icon="🔍")

# --- データベースの存在チェック ---
db_exists = os.path.exists(DB_FILE)

# ==============================================================================
# 🗂️ サイドバー（アプリ説明・案内）
# ==============================================================================
st.sidebar.title("🚀About App")
st.sidebar.markdown("""
**Japanese WordNet Search** は、概念辞書「WordNet」の日本語・英語データを検索できるツールです。
単語を入力すると**Synset（類義関係のセット）・類義語・概念定義・上位語や下位語、および対応する日英の類語**をまとめて検索できます。
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📖 使い方")
st.sidebar.markdown("""
1. 最初のみ、必要に応じて日本語WordNet DBをロードしてください。
2. メイン画面の検索ボックスに、調べたい単語（日本語または英語）を入力します。
   * ※ 完全一致した単語のみヒットします。
""")

# --- ロード処理の配置 ---
if not db_exists:
    st.sidebar.warning("⚠️ 日本語専用DB（wnjpn.db）が未ロードです。")
    if st.sidebar.button("📦 データベースファイルを読み込む"):
        st.sidebar.info("プロジェクト内の `wnjpn.db` を探索中...")
        # 既にファイルが存在する場合はチェックをパスしてリロード
        if os.path.exists(DB_FILE):
            st.sidebar.success("DBファイルのロードに成功しました！")
            st.rerun()
        else:
            st.sidebar.error("リポジトリ内に `wnjpn.db` が見つかりません。ファイルを配置してください。")
else:
    st.sidebar.success("✅ 日本語専用DB: ロード済み (キャッシュ有効)")

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
st.title("🔍 Japanese WordNet Search")
st.markdown("日本語の単語を入力して検索すると、WordNetでのSynset（概念）と類義語、各Synsetの上位語（親概念）や下位語（子概念）が表示されます。")

# 🔍 検索キーワード入力
word_input = st.text_input("検索したい単語を入力してください（日本語・英語両対応）：", value="", placeholder="例: 勉強、study、走る、run")

# 検索ワードが入力された場合のみ処理を実行
if word_input:
    # 双方からデータ件数を事前取得
    # 1. NLTK版の取得（日本語と英語の両方で検索をかける）
    all_synsets_nltk = wn.synsets(word_input, lang='jpn') + wn.synsets(word_input, lang='eng')
    # 重複するSynsetを除外して一意にする
    seen_synsets = set()
    unique_synsets_nltk = []
    for syn in all_synsets_nltk:
        if syn.name() not in seen_synsets:
            seen_synsets.add(syn.name())
            unique_synsets_nltk.append(syn)
    count_nltk = len(unique_synsets_nltk)
    
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
            st.metric(label="🗾日本語WordNet", value="未ロード", delta="サイドバーから有効化できます", delta_color="off")
    with col_count2:
        st.metric(label="🌍OMW", value=f"{count_nltk} 件")

    st.markdown("---")

    # ➔ 結果は2段 st.tabs で切り替え
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
                                
                                # 1. 定義 (Definitions) の取得と分離処理
                                def_ja, def_en, extra_exs = get_definitions_all_langs_db(syn_id)
                                st.markdown("**📝 概念の定義 (Definitions):**")
                                st.markdown(f"- 🇯🇵 **日本語:** {def_ja}")
                                st.markdown(f"- 🇺🇸 **英語:** *{def_en}*")
                                
                                # 2. 例文 (Examples) の統合
                                examples_en = get_examples_db(syn_id, lang="eng")
                                examples_ja = get_examples_db(syn_id, lang="jpn")
                                all_examples = list(set(examples_en + examples_ja + extra_exs))
                                
                                st.markdown("**📖 例文 (Examples):**")
                                if all_examples:
                                    for ex in all_examples: 
                                        st.markdown(f"- *{ex}*")
                                else:
                                    st.caption("例文なし")
                                
                                # 3. 類義語 (Synonyms)
                                ja_lemmas = get_lemmas_db(syn_id, lang="jpn")
                                en_lemmas = get_lemmas_db(syn_id, lang="eng")
                                
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown("**🇯🇵 日本語の類義語:**")
                                    st.info(", ".join(ja_lemmas) if ja_lemmas else "なし")
                                with c2:
                                    st.markdown("**🇺🇸 英語の類義語 (Synonyms):**")
                                    st.success(", ".join(en_lemmas) if en_lemmas else "なし")
                                    
                                # 4. 概念の親子・兄弟関係
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
        if not unique_synsets_nltk:
            st.info(f"Multilingual：『{word_input}』に一致する概念は見つかりませんでした。")
        else:
            pos_groups_nltk = {}
            for syn in unique_synsets_nltk:
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
                            
                            # 1. 定義 (Definitions)
                            st.markdown("**📝 概念の定義 (Definitions):**")
                            # OMW(NLTK)の標準定義は通常英語のみのため、識別子をつけて綺麗に出力
                            st.markdown(f"- 🇺🇸 **英語:** *{syn.definition()}*")
                            
                            # 2. 例文 (Examples)
                            examples = syn.examples()
                            st.markdown("**📖 例文 (Examples):**")
                            if examples:
                                for ex in examples: st.markdown(f"- *\"{ex}\"*")
                            else:
                                st.caption("例文なし")
                            
                            # 3. 類義語 (Synonyms)
                            ja_lemmas = syn.lemma_names('jpn')
                            en_lemmas = syn.lemma_names('eng')
                            
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**🇯🇵 日本語の類義語:**")
                                st.info(", ".join(ja_lemmas) if ja_lemmas else "なし")
                            with c2:
                                st.markdown("**🇺🇸 英語の類義語 (Synonyms):**")
                                st.success(", ".join(en_lemmas) if en_lemmas else "なし")
                                
                            # 4. 概念の親子・兄弟関係
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
    st.info("💡上の検索ボックスに調べたい単語を入力して Enter を押してください。")