# Routine: wiki refresh

Run by the wiki agent after each stage lands.

1. `pip install -r docs/wiki/_build/requirements.txt` (once), then `mkdocs build --strict`.
   Any warning is a finding.
2. Open the emitted pages in `site/` for the generated reference pages (tree map, results index,
   decisions index) and check they reflect `HEAD`.
3. Check dead links: the strict build catches in-site ones; spot-check GitHub links for files the
   stage moved or deleted.
4. Fix Guide pages the stage made wrong. Facts that belong to another file get a link, not a copy.
5. End in a merged PR (`docs/GIT_PROTOCOL.md`), and one logbook line.
