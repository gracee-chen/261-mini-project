"""
One-time setup: create ~/.kaggle/kaggle.json for API authentication.
Run: python setup_kaggle.py
Or: KAGGLE_USER=your_username KAGGLE_KEY=your_key python setup_kaggle.py
"""
import json
import os

username = os.environ.get("KAGGLE_USER") or input("Username: ").strip()
key = os.environ.get("KAGGLE_KEY") or input("API key: ").strip()

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
path = os.path.join(kaggle_dir, "kaggle.json")

with open(path, "w") as f:
    json.dump({"username": username, "key": key}, f, indent=2)
os.chmod(path, 0o600)
print(f"Saved to {path}")
