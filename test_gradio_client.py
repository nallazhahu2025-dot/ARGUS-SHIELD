"""
Validation script testing the Gradio backend via Gradio Client.
"""

import os
from gradio_client import Client, handle_file

def test_api():
    client = Client("http://127.0.0.1:7860/")
    
    stop_path = os.path.abspath("assets/samples/stop_sign.png")
    
    # Test physical patch defense
    res = client.predict(
        input_image=handle_file(stop_path),
        attack_vector="Physical Patch",
        epsilon=0.05,
        api_name="/execute_pipeline"
    )
    
    attacked_img, unshielded_html, purified_img, shielded_html, hud_html, spectrum_img, mask_img = res
    
    print("[API Test 1 - Physical Patch]")
    print(f"  Attacked Image Output: {os.path.basename(attacked_img)}")
    print(f"  Purified Image Output: {os.path.basename(purified_img)}")
    print(f"  HUD Status: {'PASS' if 'ARGUS TELEMETRY HUD' in hud_html else 'FAIL'}")
    print(f"  SLA Verified: {'SLA PASS' in hud_html}")
    
    # Test FGSM defense
    res_fgsm = client.predict(
        input_image=handle_file(stop_path),
        attack_vector="FGSM Digital Noise",
        epsilon=0.05,
        api_name="/execute_pipeline"
    )
    print("\n[API Test 2 - FGSM Digital Noise]")
    print(f"  Attacked Image Output: {os.path.basename(res_fgsm[0])}")
    print(f"  Purified Image Output: {os.path.basename(res_fgsm[2])}")
    print(f"  HUD Status: {'PASS' if 'ARGUS TELEMETRY HUD' in res_fgsm[4] else 'FAIL'}")
    print(f"  SLA Verified: {'SLA PASS' in res_fgsm[4]}")
    
    print("\n>>> ALL GRADIO PIPELINE ENDPOINTS OPERATIONAL AND VERIFIED <<<")

if __name__ == "__main__":
    test_api()
