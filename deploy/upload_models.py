"""
One-time uploader: push all trained model artifacts to a Hugging Face Hub
model repo so the deployed Streamlit Space can download them at runtime.

The Space git repo stays tiny (code + small CSVs only); the ~436 MB of models
live here instead. The app pulls them via snapshot_download when MODEL_REPO_ID
is set (see app/predictor.py).

Prerequisites:
    pip install -U huggingface_hub
    huggingface-cli login            # or: export HF_TOKEN=hf_xxx

Usage:
    python deploy/upload_models.py --repo-id <your-username>/indo-abusive-detector

Uploads (from saved_models/):
    indobert_finetuned/   -> indobert_finetuned/   (config + safetensors + tokenizer)
    lr_model.pkl, lr_tfidf.pkl, nb_model.pkl, nb_tfidf.pkl, svm_model.pkl
"""

import argparse
import os
import sys

# Conda sometimes sets SSL_CERT_FILE to a path that no longer exists, which makes
# httpx (used by huggingface_hub) crash when building its SSL context. Drop the
# stale value before any Hub call. (Same guard as pipeline/train_bert.py.)
_ssl_cert = os.environ.get("SSL_CERT_FILE")
if _ssl_cert and not os.path.isfile(_ssl_cert):
    del os.environ["SSL_CERT_FILE"]

from huggingface_hub import HfApi, create_repo

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED = os.path.join(ROOT, "saved_models")
CLASSICAL = [
    "lr_model.pkl", "lr_tfidf.pkl",
    "nb_model.pkl", "nb_tfidf.pkl",
    "svm_model.pkl",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True,
                    help="Target Hub model repo, e.g. user/indo-abusive-detector")
    ap.add_argument("--private", action="store_true",
                    help="Create the repo as private (the Space then needs an HF token).")
    args = ap.parse_args()

    bert_dir = os.path.join(SAVED, "indobert_finetuned")
    missing = [p for p in [bert_dir, *[os.path.join(SAVED, f) for f in CLASSICAL]]
               if not os.path.exists(p)]
    if missing:
        sys.exit("Missing artifacts — train first:\n  " + "\n  ".join(missing))

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"Repo ready: {args.repo_id}  (private={args.private})")

    print("Uploading indobert_finetuned/ ...")
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=bert_dir,
        path_in_repo="indobert_finetuned",
        commit_message="Add fine-tuned IndoBERTweet",
    )

    for fname in CLASSICAL:
        print(f"Uploading {fname} ...")
        api.upload_file(
            repo_id=args.repo_id,
            path_or_fileobj=os.path.join(SAVED, fname),
            path_in_repo=fname,
            commit_message=f"Add {fname}",
        )

    print(f"\nDone. Set MODEL_REPO_ID={args.repo_id} on the Space.")


if __name__ == "__main__":
    main()
