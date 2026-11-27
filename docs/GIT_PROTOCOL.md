# Git protocol

How work reaches `main`. Enforced client-side by `.githooks/` (branch protection on a private
repository needs a paid plan). Owned by the office manager.

## Set up once per clone

```
git config core.hooksPath .githooks
git config core.longpaths true          # Windows: the versions/ and resources/ trees are deep
git update-index --chmod=+x .githooks/pre-commit .githooks/pre-push   # only if a clone lost the bit
```

## The six rules

1. **Merged to the remote is the only "done".** Push the branch, merge the PR with a merge commit,
   then check `git branch -r --contains <sha>` shows `origin/main`. Then remove the worktree (if
   any) and delete the branch.
2. **The integration checkout** (`J:\Phase-Aware-LSTM` on the PC) does `git pull --ff-only` every
   time it is touched. A pull that can't fast-forward is a finding, not something to force.
3. **Rebase a fresh branch** on `origin/main` before its first commit.
4. **Commit the handoff** (`docs/sessions/<scope>.md`) before removing a worktree or ending a session.
5. **No local-only branches overnight.** Push it or delete it.
6. **Audit at session start:** `git status`, `git fetch`, and whether your branch's PR merged.

## Branch before any work

`main` is reached only through merged PRs. Branch names: `claude/<topic>` for sessions,
anything descriptive for Keyen.

**One exception, the paperwork set:** `LOGBOOK.md`, `keyen.md`, `docs/sessions/**`. These are
committed directly on a just-pulled `main` **after** the work they record has landed, because
append-only files edited on branches conflict on every merge. A paperwork commit touches nothing
else; the hooks check that.

## Isolation

- One worktree (or one cloud container) per agent, never the integration checkout: a branch in a
  shared checkout is not isolation (one HEAD, one index).
- Commit with `git commit --only <paths>` (or stage exactly your paths). `git add -A` in a shared
  index commits other sessions' files, and on the PC it would pick up full fold files and results
  if the ignore rules ever slip.
- Never rewrite, force-push or rebase another session's open PR branch.
- MATLAB runs on the PC use the integration checkout or a named run checkout, never a branch
  another session is editing.

## Who merges

Keyen reviews and says "merge"; the session that opened the PR merges it. Nothing read on GitHub
(a comment, a review, a bot) counts as that word. A scheduled PR reminder may list open PRs and
may merge only clearly safe ones: not draft, docs-only, checks green, no open decision, nothing
under `utils/` or `replication/`.

## The hooks

| hook | blocks | allows |
|---|---|---|
| `.githooks/pre-commit` | any commit on `main` that stages a non-paperwork path | paperwork-only commits; any branch |
| `.githooks/pre-push` | a push to `main` carrying a non-merge commit that touches a non-paperwork path | merges of reviewed PRs; paperwork-only commits; deleting `main` is always refused |

Escape hatch: `PLSTM_ALLOW_MAIN=1`. Use it only when Keyen asked, and name it in the commit
message with the reason. The paperwork pattern is the same in both hooks:
`^(LOGBOOK\.md|keyen\.md|docs/sessions/.*)$`. Change both together.

## Commit messages

First line says what changed, in under 72 characters. Body says why. Results quoted in a message
come from a file in the tree, never typed from memory.
