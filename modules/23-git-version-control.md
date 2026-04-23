# Module 23 — Git version control

Your benchmark scripts and configs change constantly. Git tracks every change so you can undo mistakes and collaborate.

---

## 1. Core workflow

```bash
git status                     # what changed?
git diff                       # show exact line changes (unstaged)
git add benchmark_cooldown.py  # stage a file for commit
git commit -m "Add cooldown timer between models"  # save snapshot
git log --oneline              # history of commits
```

---

## 2. Essential commands

| Command | What it does |
|---------|-------------|
| `git init` | Create a new repo in current directory |
| `git clone <url>` | Download a repo |
| `git status` | Show modified/staged/untracked files |
| `git diff` | Show unstaged changes line-by-line |
| `git diff --staged` | Show staged changes (about to be committed) |
| `git add <file>` | Stage a file for next commit |
| `git add .` | Stage all changed files |
| `git commit -m "message"` | Save staged changes as a commit |
| `git log` | Show commit history |
| `git log --oneline` | Compact history (one line per commit) |
| `git show <hash>` | Show a specific commit's changes |

---

## 3. Branches — work without breaking main

```bash
git branch                     # list branches
git branch feature-int8        # create new branch
git checkout feature-int8      # switch to it
# or combined:
git checkout -b feature-int8   # create + switch

# Make changes, commit...
git add .
git commit -m "Add INT8 quantization for YOLO"

# Merge back to main
git checkout main
git merge feature-int8
```

---

## 4. Stash — save work temporarily

```bash
# You're mid-edit but need to switch branches
git stash                      # save uncommitted changes
git checkout main              # switch to main
# ... do something else ...
git checkout feature-int8      # come back
git stash pop                  # restore your saved changes
```

---

## 5. Remote (GitHub / GitLab)

```bash
git remote -v                  # show remote URLs
git pull                       # download + merge remote changes
git push                       # upload your commits to remote
git push -u origin main        # first push: set upstream
```

---

## 6. `git blame` — who changed this line?

```bash
git blame benchmark_cooldown.py
# Shows: commit hash, author, date, line content
# Useful when a benchmark suddenly breaks — find which commit caused it
```

---

## 7. `.gitignore` — don't track large/sensitive files

```
# .gitignore for ML projects (when NOT using Git LFS for models)
*.onnx
*.onnx.data
*.bin
*.safetensors
*.pth
*.pt
__pycache__/
.migraphx_cache/
results/
*.log
.env
```

**Never commit model weights directly to git** — they're too large and bloat history.

**Choose one approach:**
- **Option A (most projects):** `.gitignore` the weights. Store models elsewhere (HuggingFace Hub, shared drive, cloud).
- **Option B (Git LFS):** If models *must* live in the repo, remove `*.onnx` from `.gitignore` and use LFS instead (see below).

---

## 8. Git LFS — large file storage (alternative to .gitignore)

If ONNX models must be version-controlled in the repo, **remove `*.onnx` from `.gitignore`** and use LFS:

```bash
git lfs install                      # one-time setup
git lfs track "*.onnx"               # track ONNX files with LFS
git add .gitattributes               # commit the tracking config
git add model.onnx                   # works because *.onnx is NOT in .gitignore
git commit -m "Add model via LFS"
git push
```

HuggingFace Hub uses Git LFS for all model weights.

---

## 9. Common recovery

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard all uncommitted changes (DANGEROUS)
git checkout -- .

# Restore a single file from last commit
git checkout -- benchmark_cooldown.py
```

---

## Module 23 checklist

- [ ] Can run `git status` + `git diff` to see what changed
- [ ] Can create a commit with a descriptive message
- [ ] Can create and switch branches
- [ ] Can use `git stash` to save and restore work
- [ ] Know what goes in `.gitignore` for an ML project
- [ ] Can explain why model weights shouldn't be committed to git

**Next:** `24-process-monitoring.md`
