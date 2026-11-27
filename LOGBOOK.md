# Logbook

The one record of what was done and found. **Append-only:** a wrong entry is corrected by a later
entry, never rewritten. A decision gets one line here and lives in full in `keyen.md`. Nothing here
may contradict `CLAUDE.md` or `keyen.md`; if it does, they win and the next entry says so.

**Search it, don't read it whole** (`grep -n "<term>" LOGBOOK.md`).

Format: `## yyyy-mm-dd · scope · topic`, then a few lines: what was done, what was found, where the
evidence is (file, PR, commit).

## Before this logbook

History before 2026-11-28 is not copied in. It lives in:

- git history and merged PRs 1-4, 8, 9, 12 (`git log --merges`);
- the result notes in `replication/results/`, `replication/sourceReplication/README.md`,
  `replication/versions/README.md`;
- the claude.ai project's memory and files (project notes, plans, fact checks).

---

## 2026-11-28 · office manager · workflow adopted

Adopted the graph-asr workflow (chassis, contract, records, four-scope squad, routines, MkDocs
wiki). Decision in `keyen.md` (2026-11-28). Hooks fixed against the source doc's known defects:
the paperwork regex no longer lists `docs/requests/`, and both hooks are stored executable.
