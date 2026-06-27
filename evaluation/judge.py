import json
import httpx
from langchain_anthropic import ChatAnthropic

_http_client = httpx.Client(verify=False)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_anthropic_api_key

JUDGE_MODEL = "claude-sonnet-4-6"

JUDGE_PROMPT = """あなたはパデルコーチングAIの回答品質を評価する専門家です。
以下の質問・参照メモ・回答をもとに、3つの軸でそれぞれ1〜5点でスコアを付けてください。

【質問】
{question}

【参照メモ（RAG検索結果）】
{retrieved_docs}

【回答】
{answer}

【評価軸】
- relevance（関連性）: RAG検索結果と回答の一致度（1=全く一致しない, 5=完全に一致）
- faithfulness（忠実性）: 回答が検索メモの内容に基づいているか・捏造がないか（1=捏造多い, 5=完全に根拠あり）
- usefulness（有用性）: コーチへの相談として実用的な回答か（1=役立たない, 5=非常に有用）

返答形式（JSONのみ。説明不要）:
{{"relevance": X, "faithfulness": X, "usefulness": X, "comment": "簡潔な評価コメント"}}
"""


def judge_answer(question: str, retrieved_docs: list, answer: str) -> dict:
    """
    LLM as a Judge: GPT-4oで回答品質を3軸評価する。
    人間の評価者に近い判断を自動化することで、大量のテストケースを効率的に評価できる。
    """
    api_key = get_anthropic_api_key()
    llm = ChatAnthropic(model=JUDGE_MODEL, temperature=0, anthropic_api_key=api_key, http_client=_http_client)

    docs_text = "\n\n".join(
        f"[{d['tool']}]\n{d['result']}" for d in retrieved_docs
    ) if retrieved_docs else "（検索結果なし）"

    prompt = JUDGE_PROMPT.format(
        question=question,
        retrieved_docs=docs_text,
        answer=answer,
    )

    raw = llm.invoke(prompt).content.strip()

    # JSON部分のみ抽出（LLMが余分なテキストを含む場合の対策）
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {"relevance": 0, "faithfulness": 0, "usefulness": 0, "comment": "解析失敗"}

    return scores
