# Linking & posting README — read this before you publish

This file is the playbook for *how the pieces connect*. The goal: one URL collects all the SEO authority, and every social surface funnels attention to your profile and that URL.

---

## Step 0 — Fill in your one link

Once you publish the canonical copy (see below), paste its URL here and use it everywhere:

```
DEVTO_URL  = ____________________________   (your canonical home)
MEDIUM_URL = ____________________________   (secondary, points canonical → Dev.to)
```

Then do a find-and-replace for `[LINK]` and `[DEVTO_URL]` in:
- linkedin-feed-post.md
- x-thread.md

---

## What "canonical" means (the 30-second version)

The same article on Dev.to + Medium + Peerlist looks like *duplicate content* to Google, which can bury all of them. The fix is the **canonical URL**: you nominate ONE published page as the original, and every other copy adds a `canonical_url` field pointing back to it. All the search authority then pools into that one page instead of competing.

**Your canonical home = Dev.to.** Reasons: it indexes fast, has a native `canonical_url` field, ranks for developer searches, and recruiters/engineers actually browse it. Everything else points back to it.

> `article-canonical.md` is your **master text** — the source of truth you paste from. It is not a platform. Edit a fact there once, then re-sync the copies.

---

## The chronology (copy/paste checklist)

### Day 1 — Tuesday, ~09:00 your local time

- [ ] **1. Dev.to** — publish FIRST. In the front matter set `canonical_url` to the Dev.to post's own URL. Upload `diagram-disconnect-resume.png` as the cover. → this URL becomes `DEVTO_URL`.
- [ ] **2. Medium** — publish, set `canonical_url = DEVTO_URL` (Settings → "Advanced settings" → Custom canonical link when importing, or use Medium's "Import a story" from the Dev.to URL). Add the diagram as cover.
- [ ] **3. LinkedIn article** (native long-form) — publish. LinkedIn ignores canonical tags, so add a first line: *"Originally published on Dev.to →"* linking to `DEVTO_URL`. Add the diagram as the cover image.

### Day 1 — ~90 minutes later

- [ ] **4. LinkedIn feed post** — post the ~150-word version.
  - Put `DEVTO_URL` (or your LinkedIn article URL) in the **FIRST COMMENT, not the body.** LinkedIn suppresses reach on posts with outbound links in the body.
  - In the post body write: *"Full write-up + code in the comments 👇"*.
  - Attach `diagram-disconnect-resume.png` directly to the post (native images outperform link previews).
  - Within the first hour, reply to your own comment once — early engagement trains the algorithm to push it.

### Day 2 — Wednesday, ~09:00–10:00

- [ ] **5. X / Twitter thread** — post all 8 tweets. Link ONLY in the last tweet (`DEVTO_URL` or Medium). Mid-thread links cut reach.
  - **Pin the thread** to your profile for a week.
  - Reply to the last tweet ~30 min later with the diagram image for a second surface.

### Day 3–4

- [ ] **6. Peerlist** — post linking to `DEVTO_URL` or `MEDIUM_URL`. Set canonical to Dev.to if the field exists.

### Day 7

- [ ] **7. Recycle** — re-share the LinkedIn post and re-up the X thread with a fresh one-liner, e.g.: *"Posted this a week ago — the #1 question was about atomic writes, so here's the why 👇"*. Reusing a proven post is normal and effective.

---

## The two rules that matter most

1. **One canonical URL. Everything points to it.** (Dev.to.)
2. **Links go where the algorithm allows:** LinkedIn → first comment; X → last tweet; everywhere else → in the body is fine.

---

## Where each file goes (quick map)

| File | Platform | Link placement |
|---|---|---|
| `devto-medium-version.md` | Dev.to (canonical), Medium, Peerlist | body, canonical_url set |
| `linkedin-article.md` | LinkedIn native article | "Originally on Dev.to →" at top |
| `linkedin-feed-post.md` | LinkedIn feed | **first comment** |
| `x-thread.md` | X / Twitter | **last tweet only** |
| `diagram-disconnect-resume.png` | cover/attachment everywhere | — |
| `article-canonical.md` | master text — don't publish as-is | — |
