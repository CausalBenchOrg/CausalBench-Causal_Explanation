# test_invoke.py
import json
from lambda_function import handler   # adjust module name if needed

class DummyContext:
    function_name = "Causal_Explanation_local"
    memory_limit_in_mb = 512
    invoked_function_arn = "arn:aws:lambda:local"
    aws_request_id = "local-1234"

if __name__ == "__main__":
    with open("event.json", "r", encoding="utf-8") as f:
        event = json.load(f)
    ctx = DummyContext()
    resp = handler(event, ctx)
    print("=== RESPONSE ===")
    print(json.dumps(resp, indent=2, ensure_ascii=False))
