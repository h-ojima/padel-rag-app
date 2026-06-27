import httpx
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import Tool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CHROMA_DIR, CATEGORIES, CATEGORY_DESCRIPTIONS,
    EMBEDDING_MODEL, RETRIEVAL_K, get_openai_api_key,
)

_http_client = httpx.Client(verify=False)


def _search_collection(category: str, query: str) -> str:
    api_key = get_openai_api_key()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key, http_client=_http_client)
    vectorstore = Chroma(
        collection_name=category,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    results = vectorstore.similarity_search_with_score(query, k=RETRIEVAL_K)

    if not results:
        return f"[{category}] 関連するメモは見つかりませんでした。"

    output_parts = [f"【{CATEGORY_DESCRIPTIONS[category]}のメモ】"]
    for doc, score in results:
        output_parts.append(f"（類似度: {1 - score:.2f}）\n{doc.page_content}")

    return "\n\n".join(output_parts)


def get_tools() -> list[Tool]:
    tools = []
    tool_descriptions = {
        "stroke": "ストローク（フォアハンド・バックハンド・スライス・フォームライン・レボテ・ドブレ）に関するレッスンメモを検索する",
        "volley": "ボレーやネットプレー・ポジショニング・タイミングに関するレッスンメモを検索する",
        "smash": "スマッシュやレマテ（オーバーヘッドショット）に関するレッスンメモを検索する",
        "bandeja": "バンデッハ（守備的なスマッシュ）の技術・フォームに関するレッスンメモを検索する",
        "vibora": "ビボラ（攻撃的なスマッシュ）の技術・フォームに関するレッスンメモを検索する",
        "other_shot": "チキータ・サービス・ロブなどその他ショットに関するレッスンメモを検索する",
        "strategy": "試合全体の戦い方・サービス戦略・ペアとの役割分担など戦略に関するレッスンメモを検索する",
        "tactics": "ポイント単位の判断・配球パターン・状況別対応など戦術に関するレッスンメモを検索する",
    }

    for category in CATEGORIES:
        def make_search_func(cat: str):
            return lambda q: _search_collection(cat, q)

        tools.append(
            Tool(
                name=f"{category}_search",
                description=tool_descriptions[category],
                func=make_search_func(category),
            )
        )

    return tools
