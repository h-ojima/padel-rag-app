import json
import uuid
from datetime import datetime
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CHROMA_DIR, MEMOS_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    get_openai_api_key,
)
from rag.classifier import classify_memo


def _get_vectorstore(category: str) -> Chroma:
    """
    カテゴリ別にChromaDBのCollectionを取得する。
    カテゴリ分割により同一ベクトル空間に全カテゴリが混在するのを防ぎ、
    検索精度とノイズ低減を実現する。
    """
    api_key = get_openai_api_key()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    return Chroma(
        collection_name=category,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def add_memo(memo_text: str) -> str:
    """
    メモを分類→チャンク分割→Embedding→ChromaDB保存の順で処理する。
    またStreamlit Cloud再起動時の再インデックス用にJSONでも二重保存する。
    """
    category = classify_memo(memo_text)

    # チャンク分割: 長いメモを意味のある単位に分けることで検索精度を上げる。
    # RecursiveCharacterTextSplitterは段落→文→文字の順で自然な区切りを優先する。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )
    chunks = splitter.split_text(memo_text)

    memo_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # ChromaDBにEmbeddingを保存
    vectorstore = _get_vectorstore(category)
    metadatas = [
        {"memo_id": memo_id, "category": category, "created_at": timestamp, "chunk_index": i}
        for i in range(len(chunks))
    ]
    vectorstore.add_texts(texts=chunks, metadatas=metadatas)

    # Streamlit Cloudはエフェメラルストレージのため再起動でChromaDBがリセットされる。
    # JSONに二重保存しておき、起動時に自動再インデックスすることで永続性を担保する。
    memo_record = {
        "memo_id": memo_id,
        "category": category,
        "text": memo_text,
        "created_at": timestamp,
    }
    memo_file = MEMOS_DIR / f"{memo_id}.json"
    memo_file.write_text(json.dumps(memo_record, ensure_ascii=False, indent=2), encoding="utf-8")

    return category


def reindex_from_json():
    """
    Streamlit Cloud再起動後にChromaDBが空になった場合、
    JSONバックアップから全メモを再インデックスする。
    """
    memo_files = list(MEMOS_DIR.glob("*.json"))
    if not memo_files:
        return 0

    # カテゴリごとにまとめてChromaDBへ追加（API呼び出し効率化）
    from collections import defaultdict
    category_memos: dict = defaultdict(list)

    for memo_file in memo_files:
        try:
            record = json.loads(memo_file.read_text(encoding="utf-8"))
            category_memos[record["category"]].append(record)
        except Exception:
            continue

    api_key = get_openai_api_key()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )

    total = 0
    for category, records in category_memos.items():
        vectorstore = Chroma(
            collection_name=category,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        # 既存データがある場合はスキップ（重複インデックスを防ぐ）
        if vectorstore._collection.count() > 0:
            continue

        for record in records:
            chunks = splitter.split_text(record["text"])
            metadatas = [
                {
                    "memo_id": record["memo_id"],
                    "category": category,
                    "created_at": record["created_at"],
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ]
            vectorstore.add_texts(texts=chunks, metadatas=metadatas)
            total += 1

    return total
