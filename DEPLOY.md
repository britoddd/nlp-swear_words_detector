# Deploying to Hugging Face Spaces

The app is a Streamlit front end over 4 models. The models (~436 MB) are too big
for a normal git repo, so they live in a **Hub model repo** and the app downloads
them at runtime. The **Space** repo holds only code + two small CSVs.

```
HF Hub model repo  (britoddd/indo-abusive-detector)   ← 436 MB of models
        ▲ snapshot_download() at runtime
        │
HF Space  (britoddd/indo-abusive-detector-app)        ← code + archive CSVs
```

## Prerequisites

```bash
pip install -U huggingface_hub
huggingface-cli login          # paste a token with WRITE scope (hf.co/settings/tokens)
```

## 1. Upload the models to the Hub (one time, re-run after retraining)

From the project root, with the trained artifacts present in `saved_models/`:

```bash
python deploy/upload_models.py --repo-id britoddd/indo-abusive-detector
```

(Use `--private` if you want a private model repo — the Space will then also need
an `HF_TOKEN` secret to read it.)

## 2. Create the Space

On https://huggingface.co/new-space:
- **SDK:** Streamlit
- **Hardware:** CPU basic (free) is enough — BERT runs on CPU here.
- Name it e.g. `indo-abusive-detector-app`.

## 3. Tell the Space where the models are

In the Space → **Settings → Variables and secrets**, add a **variable**:

```
MODEL_REPO_ID = britoddd/indo-abusive-detector
```

(If the model repo is private, also add a **secret** `HF_TOKEN` with a read token.)

## 4. Push the app to the Space

The Space metadata header (`sdk`, `app_file: app/app.py`, …) is already at the top
of `README.md`, so you can push this repo straight to the Space:

```bash
git remote add space https://huggingface.co/spaces/britoddd/indo-abusive-detector-app
git push space master:main
```

`saved_models/` and `outputs/` are gitignored, so nothing large is pushed. The
Space builds from `requirements.txt`, runs `app/app.py`, and on first load pulls
the models from the Hub.

## What gets shipped to the Space

| Included                                   | Excluded (gitignored)        |
|--------------------------------------------|------------------------------|
| `app/`, `pipeline/`                        | `saved_models/*` (→ Hub)     |
| `archive/abusive.csv`, `new_kamusalay.csv` | `outputs/*`                  |
| `requirements.txt`, `README.md`            | `__pycache__`, dotfiles      |

## Local run (unchanged)

Without `MODEL_REPO_ID` set, the app loads models from local `saved_models/`:

```bash
streamlit run app/app.py
```

## Notes / gotchas

- **`archive/*.csv` must be pushed.** They are needed at runtime (slang + abusive
  lexicons). `.gitignore` ignores dotfiles and `saved_models`/`outputs` only, so
  the CSVs are tracked — confirm with `git ls-files archive`.
- **First load is slow** (downloads 436 MB + NLTK `punkt`). Subsequent loads hit
  the Space's cache. `@st.cache_resource` keeps models warm between requests.
- **scikit-learn is pinned to 1.7.1** to match the pickles. If you retrain with a
  different version, bump it in `requirements.txt`.
- **Retraining?** Re-run step 1 to refresh the Hub repo; the Space needs no
  redeploy (it pulls the latest on next cold start / restart).
