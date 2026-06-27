import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import httpx
_orig_client_init = httpx.Client.__init__
def _no_verify_client(self, *args, **kwargs):
    kwargs['verify'] = False
    _orig_client_init(self, *args, **kwargs)
httpx.Client.__init__ = _no_verify_client

_orig_async_client_init = httpx.AsyncClient.__init__
def _no_verify_async_client(self, *args, **kwargs):
    kwargs['verify'] = False
    _orig_async_client_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = _no_verify_async_client

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CATEGORIES, CATEGORY_DESCRIPTIONS, LLM_MODEL, get_anthropic_api_key

CLASSIFY_PROMPT = PromptTemplate.from_template("""
あなたはパデルコーチングのメモ分類専門家です。
以下のメモを最も適切な1つのカテゴリに分類してください。

【カテゴリ一覧】
{categories}

【メモ】
{memo}

カテゴリIDのみをそのまま返してください（例: stroke）。
説明や理由は不要です。
""")


def classify_memo(memo: str) -> str:
    api_key = get_anthropic_api_key()
    llm = ChatAnthropic(model=LLM_MODEL, temperature=0, anthropic_api_key=api_key)

    categories_text = "\n".join(
        f"- {cat}: {desc}" for cat, desc in CATEGORY_DESCRIPTIONS.items()
    )

    prompt = CLASSIFY_PROMPT.format(categories=categories_text, memo=memo)
    result = llm.invoke(prompt).content.strip().lower()

    if result in CATEGORIES:
        return result
    return "other_shot"
