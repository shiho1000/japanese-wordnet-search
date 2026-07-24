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

st.set_page_config(page_title="WordNet検索ツール", layout="wide", page_icon="🚀")

# --- データベースの存在チェック ---
db_exists = os.path.exists(DB_FILE)

# --- サイドバー設定 ---
st.sidebar.header("🔧 設定")

# 選択肢の動的制御
if db_exists:
    # DBが存在すれば両方選べる
    source_option = st.sidebar.radio(
        "データソースの選択",
        ("ローカルDB (SQLite)", "NLTK (オンライン読み込み)")
    )
else:
    # DBがなければNLTK固定にし、ダウンロードボタンを置く
    st.sidebar.warning("⚠️ 日本語WordNet（ローカルDB）が未セットアップです。")
    
    # 📥 ダウンロードボタン
    if st.sidebar.button("日本語WordNet DBをダウンロードする (約200MB)"):
        with st.spinner("ダウンロード中...（数分かかる場合があります）"):
            url = "https://raw.githubusercontent.com/bond-lab/wnja/master/db/wnjpn.db.gz"
            gz_file = "wnjpn.db.gz"
            urllib.request.urlretrieve(url, gz_file)
            with gzip.open(gz_file, 'rb') as f_in:
                with open(DB_FILE, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            if os.path.exists(gz_file):
                os.remove(gz_file)
        
        st.sidebar.success("ダウンロード完了！ページを再読み込みします...")
        st.rerun() # アプリを再実行してラジオボタンを有効化する
        
    source_option = "NLTK (オンライン読み込み)"

# タイトル表示の切り替え
if source_option == "NLTK (オンライン読み込み)":
    app_title = "🌍 多言語WordNet検索 (Open Multilingual Wordnet)"
    app_caption = "NLTK経由の多言語対応WordNet。品詞別、上位語・下位語、英和類義語の同時出力に対応"
else:
    app_title = "🇯🇵 日本語WordNet検索"
    app_caption = "ローカルSQLiteデータベースから検索する日本語専用WordNet。高速かつオフラインで動作"

st.sidebar.markdown("---")
st.sidebar.header("🔍 検索ワード")
word_input = st.sidebar.text_input("検索したい単語を入力してください（完全一致のみ）：", value="学校")

# --- メインコンテンツ ---
st.title(app_title)
st.caption(app_caption)

if word_input:
    # --------------------------------------------------------------------------
    # 🔥 選ばれたデータソースに応じて処理を分ける
    # --------------------------------------------------------------------------
    if source_option == "NLTK (オンライン読み込み)":
        # --- NLTK版のデータ処理 ---
        all_synsets_nltk = wn.synsets(word_input, lang='jpn')
        
        if not all_synsets_nltk:
            st.sidebar.error(f"「{word_input}」に一致する概念が見つかりませんでした。")
            st.warning(f"「{word_input}」に一致する概念が見つかりませんでした。")
        else:
            st.sidebar.success(f"「{word_input}」で {len(all_synsets_nltk)} 件の概念が見つかりました。")
            
            pos_groups = {}
            for syn in all_synsets_nltk:
                pos = syn.pos()
                if pos not in pos_groups: pos_groups[pos] = []
                pos_groups[pos].append(syn)
            
            tabs = st.tabs([POS_MAP.get(pos, pos) for pos in pos_groups.keys()])
            
            for tab, (pos, synsets) in zip(tabs, pos_groups.items()):
                with tab:
                    st.write(f"### 「{word_input}」 の検索結果: {POS_MAP.get(pos, pos)} {len(synsets)} 件")
                    for syn in synsets:
                        with st.container(border=True):
                            st.markdown(f"#### 💡 概念 ID: `{syn.name()}`")
                            st.markdown(f"**📝 英語の定義 (Definition):**  \n*{syn.definition()}*")
                            
                            examples = syn.examples()
                            if examples:
                                st.markdown("**📖 例文 (Examples):**")
                                for ex in examples: st.markdown(f"- *\"{ex}\"*")
                            
                            ja_lemmas = syn.lemma_names('jpn')
                            en_lemmas = syn.lemma_names('eng')
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**🇯🇵 日本語の類義語:**")
                                st.info(", ".join(ja_lemmas) if ja_lemmas else "なし")
                            with col2:
                                st.markdown("**🇺🇸 英語の類義語 (Synonyms):**")
                                st.success(", ".join(en_lemmas) if en_lemmas else "なし")
                                
                            st.markdown("**🌿 概念の親子・兄弟関係 (Hierarchies & Sister Terms):**")
                            rel_col1, rel_col2, rel_col3 = st.columns(3)

                            hypernyms = syn.hypernyms()
                            with rel_col1:
                                with st.expander(f"🔺 上位語 / 親概念 ({len(hypernyms)}件)"):
                                    if hypernyms:
                                        for hyper in hypernyms:
                                            st.markdown(f"• **`{hyper.name()}`**")
                                            st.caption(f"└ 日: {', '.join(hyper.lemma_names('jpn')[:5])} / 英: {', '.join(hyper.lemma_names('eng')[:5])}")
                                    else: st.write("最上位の概念です。")                                    
                            with rel_col2:
                                siblings = []
                                for hyper in hypernyms:
                                    for sibling in hyper.hyponyms():
                                        if sibling != syn and sibling not in siblings: siblings.append(sibling)
                                with st.expander(f"🔹 兄弟語 / 同階層 ({len(siblings)}件)"):
                                    if siblings:
                                        for sib in siblings:
                                            st.markdown(f"• **`{sib.name()}`**")
                                            st.caption(f"└ 日: {', '.join(sib.lemma_names('jpn')[:5])} / 英: {', '.join(sib.lemma_names('eng')[:5])}")
                                    else: st.write("兄弟概念が見つからないか、最上位概念です。")
                            with rel_col3:
                                hyponyms = syn.hyponyms()
                                with st.expander(f"🔻 下位語 / 子概念 ({len(hyponyms)}件)"):
                                    if hyponyms:
                                        for hypo in hyponyms:
                                            st.markdown(f"• **`{hypo.name()}`**")
                                            st.caption(f"└ 日: {', '.join(hypo.lemma_names('jpn')[:5])} / 英: {', '.join(hypo.lemma_names('eng')[:5])}")
                                    else: st.write("最下位の概念です。")

    else:
        # --- ローカルDB (SQLite) 版のデータ処理 ---
        all_synsets_db = get_synsets_by_word_db(word_input)
        
        if not all_synsets_db:
            st.sidebar.error(f"「{word_input}」に一致する概念が見つかりませんでした。")
            st.warning(f"「{word_input}」に一致する概念が見つかりませんでした。")
        else:
            st.sidebar.success(f"「{word_input}」で {len(all_synsets_db)} 件の概念が見つかりました。")
            
            pos_groups = {}
            for syn in all_synsets_db:
                pos = syn['pos']
                if pos not in pos_groups: pos_groups[pos] = []
                pos_groups[pos].append(syn)
            
            tabs = st.tabs([POS_MAP.get(pos, pos) for pos in pos_groups.keys()])
            
            for tab, (pos, synsets) in zip(tabs, pos_groups.items()):
                with tab:
                    st.write(f"### 「{word_input}」 の検索結果: {POS_MAP.get(pos, pos)} {len(synsets)} 件")
                    for syn in synsets:
                        syn_id = syn['id']
                        syn_name = syn['name']
                        
                        with st.container(border=True):
                            st.markdown(f"#### 💡 概念 ID: `{syn_name}`")
                            
                            definition = get_definition_db(syn_id, lang="eng")
                            st.markdown(f"**📝 英語の定義 (Definition):**  \n*{definition}*")
                            
                            examples = get_examples_db(syn_id, lang="eng")
                            if examples:
                                st.markdown("**📖 例文 (Examples):**")
                                for ex in examples: st.markdown(f"- *\"{ex}\"*")
                            
                            ja_lemmas = get_lemmas_db(syn_id, lang="jpn")
                            en_lemmas = get_lemmas_db(syn_id, lang="eng")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**🇯🇵 日本語の類義語:**")
                                st.info(", ".join(ja_lemmas) if ja_lemmas else "なし")
                            with col2:
                                st.markdown("**🇺🇸 英語の類義語 (Synonyms):**")
                                st.success(", ".join(en_lemmas) if en_lemmas else "なし")
                                
                            st.markdown("**🌿 概念の親子・兄弟関係 (Hierarchies & Sister Terms):**")
                            rel_col1, rel_col2, rel_col3 = st.columns(3)

                            hypernyms = get_related_synsets_db(syn_id, "hype")
                            with rel_col1:
                                with st.expander(f"🔺 上位語 / 親概念 ({len(hypernyms)}件)"):
                                    if hypernyms:
                                        for hyper in hypernyms:
                                            h_ja = get_lemmas_db(hyper['id'], lang="jpn")
                                            h_en = get_lemmas_db(hyper['id'], lang="eng")
                                            st.markdown(f"• **`{hyper['name']}`**")
                                            st.caption(f"└ 日: {', '.join(h_ja[:5])} / 英: {', '.join(h_en[:5])}")
                                    else: st.write("最上位の概念です。")
                                    
                            with rel_col2:
                                siblings = []
                                sibling_ids = set()
                                for hyper in hypernyms:
                                    cand_siblings = get_related_synsets_db(hyper['id'], "hypo")
                                    for sib in cand_siblings:
                                        if sib['id'] != syn_id and sib['id'] not in sibling_ids:
                                            siblings.append(sib)
                                            sibling_ids.add(sib['id'])
                                            
                                with st.expander(f"🔹 兄弟語 / 同階層 ({len(siblings)}件)"):
                                    if siblings:
                                        for sib in siblings:
                                            sib_ja = get_lemmas_db(sib['id'], lang="jpn")
                                            sib_en = get_lemmas_db(sib['id'], lang="eng")
                                            st.markdown(f"• **`{sib['name']}`**")
                                            st.caption(f"└ 日: {', '.join(sib_ja[:5])} / 英: {', '.join(sib_en[:5])}")
                                    else: st.write("兄弟概念が見つからないか、最上位概念です。")
                                    
                            with rel_col3:
                                hyponyms = get_related_synsets_db(syn_id, "hypo")
                                with st.expander(f"🔻 下位語 / 子概念 ({len(hyponyms)}件)"):
                                    if hyponyms:
                                        for hypo in hyponyms:
                                            hypo_ja = get_lemmas_db(hypo['id'], lang="jpn")
                                            hypo_en = get_lemmas_db(hypo['id'], lang="eng")
                                            st.markdown(f"• **`{hypo['name']}`**")
                                            st.caption(f"└ 日: {', '.join(hypo_ja[:5])} / 英: {', '.join(hypo_en[:5])}")
                                    else: st.write("最下位の概念です。")