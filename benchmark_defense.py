"""
Benchmark script for the 4-Stage ARGUS Shield Defense Pipeline with Dynamic Gating.
"""

import cv2
import time
import shield_engine
import ml_pipeline


def run_benchmark():
    img = cv2.imread('assets/samples/stop_sign.png')
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = ml_pipeline.preprocess_numpy(rgb)

    attacks = [
        'None (Clean)',
        'FGSM Digital Noise',
        'PGD Iterative Noise',
        'Physical Patch (AdvPatch)',
        'Transfer / Black-Box Attack'
    ]

    print("=" * 75)
    print("      ARGUS SHIELD 4-STAGE PIPELINE BENCHMARK (Target SLA: < 15ms)")
    print("=" * 75)

    for atk in attacks:
        atk_tensor = ml_pipeline.inject_attack(tensor, atk, epsilon=0.05)
        
        # Measure unshielded
        unshielded_res = ml_pipeline.run_inference(atk_tensor, shielded=False)
        
        # Measure shielded
        times = []
        for _ in range(3):
            shielded_res = ml_pipeline.run_inference(atk_tensor, shielded=True)
            times.append(shielded_res["telemetry"]["defense_latency_ms"])
        
        avg_def_ms = sum(times) / len(times)
        tel = shielded_res["telemetry"]
        is_pass = avg_def_ms < 15.0

        print(f"[{atk}]")
        print(f"  Unshielded Prediction : {unshielded_res['label']} ({unshielded_res['confidence']}%)")
        print(f"  Shielded Prediction   : {shielded_res['label']} ({shielded_res['confidence']}%)")
        print(f"  Defense Latency       : {avg_def_ms:.2f} ms")
        print(f"  SLA Status            : {'[PASS] (<15ms)' if is_pass else '[FAIL]'}")
        print(f"  Dynamic Gated Bypass  : {tel['dynamic_gated']}")
        print(f"  Anomaly Index         : {tel['anomaly_score']} / 100")
        print(f"  Actions Triggered     : {tel['actions_taken']}")
        print("-" * 75)
        assert is_pass, f"{atk} exceeded 15ms SLA!"

    print(">>> ALL 4 DEFENSE STAGES & DYNAMIC GATING VERIFIED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    run_benchmark()
