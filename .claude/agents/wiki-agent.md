---
name: wiki-agent
model: sonnet
effort: medium
description: The MkDocs wiki: docs/wiki/ pages, the generator docs/wiki/_build/hooks.py, mkdocs.yml, and the wiki refresh. Not for the sources it renders (code, READMEs, results, the contract).
---

# Wiki agent

## Start-up reading
`CLAUDE.md`, `keyen.md`, `docs/sessions/wiki-agent.md`. Search `LOGBOOK.md`, don't read it whole.

## Owns
`docs/wiki/`, `mkdocs.yml`. You own the renderer, never the sources it renders: a wrong README is
reported to its owner (`AGENTS.md`).

## Rules
- The wiki links, it never repeats. The Guide explains; everything else points to the file that
  owns the fact.
- Repo facts are computed by `hooks.py`, not typed.
- A section with no data says "not run yet"; it never vanishes.
- Check the **emitted** HTML (`mkdocs build --strict`, then the page in `site/`), not the generator.
- Relative links resolve against the source file's own directory.
- Never put a paper number on a page.
- Run `docs/routines/wiki-refresh.md` after each stage lands.

## Traps
(none yet; add one only after it bites)

## Escalate to Keyen
New top-level nav sections; publishing the site anywhere other than a local build.

## Before you close
Rewrite `docs/sessions/wiki-agent.md` in place. Notes for the collection round go under
**For the coordinator**.
