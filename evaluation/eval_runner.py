"""
ローカル実行専用の評価スクリプト。本番アプリには組み込まない。

使い方:
    python evaluation/eval_runner.py

評価対象: data/histories/ 以下の全会話履歴
結果: evaluation/eval_results_YYYYMMDD.json に保存
"""
import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import HISTORIES_DIR
from evaluation.judge import judge_answer


def run_evaluation():
    history_files = list(HISTORIES_DIR.glob("*.json"))
    if not history_files:
        print("評価対象の会話履歴がありません。")
        return

    results = []
    for history_file in history_files:
        record = json.loads(history_file.read_text(encoding="utf-8"))
        session_id = record.get("session_id", history_file.stem)
        messages = record.get("messages", [])

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            question = ""
            if i > 0 and messages[i - 1]["role"] == "user":
                question = messages[i - 1]["content"]

            answer = msg.get("content", "")
            retrieved_docs = msg.get("retrieved_docs", [])

            if not question or not answer:
                continue

            print(f"評価中: {session_id} - {question[:30]}...")
            scores = judge_answer(question, retrieved_docs, answer)

            results.append({
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "scores": scores,
            })

    output_path = Path(__file__).parent / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if results:
        avg_relevance = sum(r["scores"].get("relevance", 0) for r in results) / len(results)
        avg_faithfulness = sum(r["scores"].get("faithfulness", 0) for r in results) / len(results)
        avg_usefulness = sum(r["scores"].get("usefulness", 0) for r in results) / len(results)
        print(f"\n評価完了: {len(results)}件")
        print(f"  関連性: {avg_relevance:.2f} / 5")
        print(f"  忠実性: {avg_faithfulness:.2f} / 5")
        print(f"  有用性: {avg_usefulness:.2f} / 5")
        print(f"結果保存: {output_path}")


if __name__ == "__main__":
    run_evaluation()
