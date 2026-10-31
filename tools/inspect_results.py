import json
import glob
import os

print("=== 1. Inspecting pilot_remote_events.jsonl ===")
p = "outputs/colab/pilot_remote_events.jsonl"
if os.path.exists(p):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        events = [json.loads(line) for line in f if line.strip()]
    print("Total events:", len(events))
    for e in events:
        st = e.get("stage")
        sub = e.get("sub_stage")
        dev_loss = e.get("dev_loss")
        if dev_loss is not None or st in ["stage1_eval", "stage2_eval", "arm_completed", "initial_eval"]:
            print(f"Stage: {st}, Sub: {sub}, Step: {e.get('step')}, DevLoss: {dev_loss}, EvDone: {e.get('eval_examples_done')}")

print("\n=== 2. Inspecting grounding_eval_it_encoder_frozen_then_joint.json ===")
p_grounding = "outputs/colab/grounding_eval_it_encoder_frozen_then_joint.json"
if os.path.exists(p_grounding):
    with open(p_grounding, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Keys in grounding eval:", list(data.keys()))
    if "conditions" in data:
        for cond, res in data["conditions"].items():
            cases = res.get("cases", [])
            app_cnt = res.get("applied_count", 0)
            p_cnt = res.get("pass_count", 0)
            print(f"Condition: {cond} -> Applied: {app_cnt}/{len(cases)}, Pass: {p_cnt}/{len(cases)}")
            reasons = {}
            for c in cases:
                r = c.get("reason", "unknown")
                reasons[r] = reasons.get(r, 0) + 1
            for r, count in reasons.items():
                print(f"    - {r[:80]}: {count}")


print("\n=== 3. Inspecting latest zip or json files ===")
for jf in glob.glob("outputs/colab/*.json"):
    print(f"File: {jf} ({os.path.getsize(jf)} bytes)")
