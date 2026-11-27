# Session handoffs

One file per scope, **rewritten in place** at the end of every session: `office-manager.md`,
`pipeline-engineer.md`, `researcher.md`, `wiki-agent.md`. Not a numbered series: a series becomes a
second log.

Every handoff has these sections:

- **Next**: what the next session in this scope should do first.
- **Settled**: what this scope has decided and must not reopen (link `keyen.md` where it decided).
- **Broken or open**: known faults and unanswered questions.
- **Already fixed, do not re-report**: so the next session doesn't rediscover them.
- **For the coordinator**: notes for the collection round (`docs/routines/collection-round.md`).

**The test for what belongs here:** if you're keeping what a previous session did because
deleting it feels wrong, it belongs in `LOGBOOK.md`, not here.

Handoffs are paperwork: commit them on a just-pulled `main` after the work they describe has
landed (`docs/GIT_PROTOCOL.md`).
