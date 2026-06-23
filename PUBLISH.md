# Publishing this repo

## 1. Repo name & visibility

**Recommended name:** `bulletproof-training`
*(memorable, matches the article; alternatives: `resumable-training`, `never-lose-a-training-run`)*

**Visibility: Public.** The whole point is discoverability — recruiters check
GitHub, and Pages only serves the interactive page from a public repo (free
tier). Make it private only if you want to polish before anyone sees it.

## 2. Create the repo and push

```bash
cd "path/to/this/folder"

git init
git add .
git commit -m "Resumable, idempotent training on ephemeral GPUs"
git branch -M main

# create an empty repo named bulletproof-training on github.com first, then:
git remote add origin https://github.com/<your-username>/bulletproof-training.git
git push -u origin main
```

## 3. Turn on GitHub Pages (for the interactive walkthrough)

1. Repo → **Settings** → **Pages**
2. **Source:** Deploy from a branch
3. **Branch:** `main`, **Folder:** `/docs`
4. Save. In ~1 minute your page is live at:
   `https://<your-username>.github.io/bulletproof-training/`
5. Put that URL in the README (replace the placeholder) and in your article CTAs.

## 4. Make it work for you

- **Pin the repo** on your GitHub profile (Profile → Customize your pins).
- Add a one-line **repo description** + the Pages URL in the repo's "About".
- Add **topics**: `machine-learning`, `pytorch`, `mlops`, `checkpointing`,
  `reproducibility`, `colab`.
- Link the Pages URL from your LinkedIn **Featured** section and the article.

## 5. A note on the `content/` folder

`content/` holds the article and every platform version + the posting and
recruiter strategy. That's intentional — it documents the writing, and the
strategy files are useful to you. If you'd rather keep the marketing playbooks
private, move `content/germany-recruiter-playbook.md`,
`content/titles-hashtags-posting-plan.md`, and `content/LINKING-README.md` out
before the first commit (or add them to `.gitignore`). The article itself
(`article-canonical.md`) is great to keep public.

## Heads-up on the original draft

`bulletproof-training-on-ephemeral-gpus.md` in the repo root is your original
draft. The polished version lives in `content/article-canonical.md`. Delete the
root draft (or move it into `content/`) before committing if you don't want both.
