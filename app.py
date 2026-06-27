import streamlit as st
from datetime import datetime

from config import CATEGORY_DESCRIPTIONS
from rag.indexer import add_memo, reindex_from_json
from chat.chain import build_agent, run_agent
from chat.history import save_history, load_history, list_histories, new_session_id

st.set_page_config(page_title="パデルAIコーチ", page_icon="🎾", layout="wide")

# --- 起動時の再インデックス ---
# Streamlit CloudはChromaDBがリセットされるため、起動のたびにJSONから復元する
if "reindexed" not in st.session_state:
    with st.spinner("メモデータを読み込み中..."):
        count = reindex_from_json()
        if count > 0:
            st.toast(f"{count}件のメモを復元しました", icon="✅")
    st.session_state.reindexed = True

# --- セッション状態の初期化 ---
if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memo_saved_label" not in st.session_state:
    st.session_state.memo_saved_label = None
if "agent" not in st.session_state:
    # AgentはLLM・Toolの初期化コストが高いためセッション内でキャッシュする
    st.session_state.agent = build_agent()

# ========== サイドバー ==========
with st.sidebar:
    st.title("🎾 パデルAIコーチ")

    # --- メモ記録セクション ---
    st.header("📝 レッスンメモを記録")
    memo_input = st.text_area(
        "今日学んだことを記録してください",
        placeholder="例: バックハンドのフォロースルーを意識すると安定した。肘を伸ばしきらずにコンパクトに振る。",
        height=150,
        key="memo_text_area",
    )
    if st.button("記録する", type="primary", use_container_width=True):
        if memo_input.strip():
            with st.spinner("メモを分析・保存中..."):
                try:
                    category = add_memo(memo_input.strip())
                    label = CATEGORY_DESCRIPTIONS.get(category, category)
                    st.session_state.memo_saved_label = label
                    st.session_state.memo_text_area = ""
                    st.rerun()
                except Exception as e:
                    st.error(f"保存に失敗しました: {e}")
        else:
            st.warning("メモを入力してください")

    if st.session_state.memo_saved_label:
        st.success(f"記録しました！（カテゴリ：{st.session_state.memo_saved_label}）")
        st.session_state.memo_saved_label = None

    st.divider()

    # --- 新しい会話ボタン ---
    if st.button("＋ 新しい会話を開始", use_container_width=True):
        if st.session_state.messages:
            save_history(st.session_state.session_id, st.session_state.messages)
        st.session_state.session_id = new_session_id()
        st.session_state.messages = []
        st.rerun()

    # --- 過去の会話一覧 ---
    st.header("💬 過去の会話")
    histories = list_histories()
    if not histories:
        st.caption("まだ会話履歴がありません")
    for h in histories:
        label = h["title"] or "（無題）"
        created = h["created_at"][:10] if h["created_at"] else ""
        if st.button(f"📄 {label}\n{created}", key=h["session_id"], use_container_width=True):
            if st.session_state.messages:
                save_history(st.session_state.session_id, st.session_state.messages)
            record = load_history(h["session_id"])
            st.session_state.session_id = h["session_id"]
            st.session_state.messages = record.get("messages", [])
            st.rerun()

# ========== メインエリア ==========
st.title("パデルAIコーチ 🎾")
st.caption("レッスンメモに基づいてパデルのアドバイスを提供します")

# 会話履歴を表示
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # アシスタントメッセージには参照メモを折りたたみ表示
        if msg["role"] == "assistant" and msg.get("retrieved_docs"):
            with st.expander("参照したメモを見る"):
                for doc in msg["retrieved_docs"]:
                    st.markdown(f"**🔍 {doc['tool']}**")
                    st.text(doc["result"])

# チャット入力
if user_input := st.chat_input("コーチに質問してください..."):
    # ユーザーメッセージを追加
    user_msg = {
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat(),
    }
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.write(user_input)

    # Agent実行
    with st.chat_message("assistant"):
        with st.spinner("メモを検索して回答を生成中..."):
            try:
                result = run_agent(
                    st.session_state.agent,
                    user_input,
                    st.session_state.messages[:-1],  # 現在のユーザー入力を除いた履歴
                )

                st.write(result["answer"])

                if result["retrieved_docs"]:
                    with st.expander("参照したメモを見る"):
                        for doc in result["retrieved_docs"]:
                            st.markdown(f"**🔍 {doc['tool']}**")
                            st.text(doc["result"])

                assistant_msg = {
                    "role": "assistant",
                    "content": result["answer"],
                    "timestamp": datetime.now().isoformat(),
                    "retrieved_docs": result["retrieved_docs"],
                }
                st.session_state.messages.append(assistant_msg)

                # 会話を自動保存
                save_history(st.session_state.session_id, st.session_state.messages)

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
