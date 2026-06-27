import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# APIキー: 開発時は.env、Streamlit Cloud本番時はst.secrets経由で取得
def get_openai_api_key() -> str:
    try:
        import streamlit as st
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")

def get_anthropic_api_key() -> str:
    try:
        import streamlit as st
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.getenv("ANTHROPIC_API_KEY", "")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
HISTORIES_DIR = DATA_DIR / "histories"
# メモをJSONでも二重保存するディレクトリ（Streamlit Cloud再起動後の再インデックス用）
MEMOS_DIR = DATA_DIR / "memos"

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
HISTORIES_DIR.mkdir(exist_ok=True)
MEMOS_DIR.mkdir(exist_ok=True)

# ChromaDB Collection名（カテゴリ別に分けることで検索精度向上・ノイズ削減）
CATEGORIES = [
    "stroke",
    "volley",
    "smash",
    "bandeja",
    "vibora",
    "other_shot",
    "strategy",
    "tactics",
]

CATEGORY_DESCRIPTIONS = {
    "stroke": "ストローク（フォアハンド・バックハンド・スライス・フォームライン・レボテ・ドブレ）",
    "volley": "ボレー（ネットプレー・ボレーのポジショニング・タイミング）",
    "smash": "スマッシュ・レマテ（オーバーヘッド・レマテの打ち方・フォーム）",
    "bandeja": "バンデッハ（守備的スマッシュ・バンデッハの技術・フォーム）",
    "vibora": "ビボラ（攻撃的スマッシュ・ビボラの技術・フォーム）",
    "other_shot": "その他ショット（チキータ・サービス・ロブ等の特殊ショット）",
    "strategy": "戦略（試合全体の戦い方・サービス戦略・ペアとの役割分担）",
    "tactics": "戦術（ポイント単位の判断・配球パターン・状況別対応）",
}

# Embedding: OpenAI text-embedding-3-smallはコスパ・精度バランスが良くRAGに最適
EMBEDDING_MODEL = "text-embedding-3-small"

# LLMモデル: temperature=0で安定した回答を生成（ランダム性を排除）
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_TEMPERATURE = 0

# チャンク設定: パデルメモの文章量（数十〜数百字）に合わせた設定
# chunk_sizeを小さくすることで意味のまとまりを保ちながら検索精度を上げる
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30  # 境界部分の文脈を失わないための重複

# 1クエリで取得するチャンク数: 多すぎると無関係な情報が混入するため3件に制限
RETRIEVAL_K = 3

SYSTEM_PROMPT = """あなたはパデル専門のAIコーチです。ユーザーが記録したレッスンメモをもとに的確なアドバイスを提供してください。

【ルール】
- 必ずToolで検索したメモに基づいて回答すること
- 記録に基づかない内容は「記録が見つかりません」と伝えること
- 回答は日本語で、コーチとして分かりやすく丁寧に伝えること
- 必要に応じて複数カテゴリのToolを使って検索すること"""
