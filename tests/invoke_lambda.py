"""
Test causal analysis by invoking the lambda function locally.
"""

import json

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lambda_function import handler


class DummyContext:
    function_name = "Causal_Explanation_local"
    memory_limit_in_mb = 512
    invoked_function_arn = "arn:aws:lambda:local"
    aws_request_id = "local-1234"


def invoke(event: str):
    with open(f"{ROOT}/tests/events/{event}.json", "r", encoding="utf-8") as f:
        event = json.load(f)
    ctx = DummyContext()
    resp = handler(event, ctx)
    print("=== RESPONSE ===")
    print(json.dumps(resp, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # event file name without extension
    event = "event1"

    # invoke the lambda function
    invoke(event)
