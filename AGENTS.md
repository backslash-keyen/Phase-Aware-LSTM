# Agents and ownership

The ownership authority. Every tracked path falls under exactly one scope; the contradiction audit
(`docs/routines/contradiction-audit.md`) checks that. Current state only; history goes to
`LOGBOOK.md`. Scope-to-scope asks go direct; anything for Keyen goes through the project's
coordinator chat.

## Scopes

| scope | agent file | owns |
|---|---|---|
| office manager | `.claude/agents/office-manager.md` | `CLAUDE.md`, `AGENTS.md`, `README.md`, `.gitignore`, `.gitattributes`, `.githooks/`, `.claude/`, `docs/GIT_PROTOCOL.md`, `docs/routines/`, `docs/sessions/README.md`, `BACKLOG.md` (layout only; Keyen orders it), `LICENSE` |
| pipeline engineer | `.claude/agents/pipeline-engineer.md` | `replication/` (all of it, including `results/`, `variants/`, `versions/`, `sourceReplication/`, `diagnostics/`), `datasets/`, `replicatePaper.m`, and custody of the released code: `utils/`, `projecitonTL_mainScript.m`, `PhaseAwareLSTM.prj`, `resources/`, `readMe.md.txt`, `readMe_fromZip.md`, `ProjectionTransferLearning_usingPhaseAwareLSTM.pdf` (custody = keeps them unchanged) |
| researcher | `.claude/agents/researcher.md` | `docs/literature/` |
| wiki agent | `.claude/agents/wiki-agent.md` | `docs/wiki/` (pages and `_build/`), `mkdocs.yml` |

Shared by everyone, append or rewrite-your-own only:

| path | rule |
|---|---|
| `LOGBOOK.md` | append-only; any scope appends |
| `keyen.md` | Keyen's; agents append to **Log** only |
| `docs/sessions/<scope>.md` | each scope rewrites its own |

## Assigned now

| scope | on |
|---|---|
| pipeline engineer | CWL 3-class replication (branch `claude/cwl-3class`); per-session z-score test paused behind it |
| office manager | this workflow adoption |
| researcher | idle |
| wiki agent | first wiki build |

## Open PR branches (don't rewrite)

PRs 5, 6, 7, 10, 11, 13, 14, 15 are open. A session may push to a PR branch only if it opened that
PR.
