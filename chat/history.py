import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import HISTORIES_DIR


def save_history(session_id: str, messages: list, title: str = "") -> Path:
    if not title and messages:
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        title = first_user[:30] + ("..." if len(first_user) > 30 else "")

    record = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "title": title,
        "messages": messages,
    }
    path = HISTORIES_DIR / f"{session_id}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_history(session_id: str) -> dict:
    path = HISTORIES_DIR / f"{session_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def list_histories() -> list[dict]:
    histories = []
    for path in sorted(HISTORIES_DIR.glob("*.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            histories.append({
                "session_id": record.get("session_id", path.stem),
                "title": record.get("title", "（無題）"),
                "created_at": record.get("created_at", ""),
            })
        except Exception:
            continue
    return histories


def new_session_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")
