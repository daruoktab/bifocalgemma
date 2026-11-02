import json
import zipfile

with zipfile.ZipFile('outputs/colab/it_encoder_frozen_then_joint_results.zip') as z:
    d = json.loads(z.read('pilot_e2b_e2b.json').decode('utf-8'))

print("=== INITIAL EVAL ===")
print(json.dumps(d.get("initial_eval"), indent=2))

print("\n=== ARMS ===")
for arm_name, arm_data in d.get("arms", {}).items():
    print(f"--- Arm: {arm_name} ---")
    print("Dev Loss:", arm_data.get("dev_loss"))
    print("Metrics summary:")
    # print any metrics like copy probes, humaneval
    for k, v in arm_data.items():
        if k not in ["stages", "history"]:
            print(f"  {k}: {v}")

print("\n=== COMPARISON TABLE ===")
print(json.dumps(d.get("comparison_table"), indent=2))
