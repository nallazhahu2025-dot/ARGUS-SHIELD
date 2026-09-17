"""
Sample benchmark images manager for ARGUS Shield.
Ensures high-fidelity benchmark images exist in assets/samples/.
"""

import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "samples")
os.makedirs(ASSETS_DIR, exist_ok=True)

REQUIRED_SAMPLES = [
    "stop_sign.png",
    "panda.png",
    "sports_car.png",
    "golden_retriever.png"
]


def main():
    for fname in REQUIRED_SAMPLES:
        path = os.path.join(ASSETS_DIR, fname)
        if os.path.exists(path):
            print(f"Verified sample asset exists: {fname}")
        else:
            print(f"Warning: {fname} not found!")


if __name__ == "__main__":
    main()
