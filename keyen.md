# Keyen

Keyen's own file. Agents read it at session start and treat it as instruction that beats anything
they inferred. Agents never reorganise it; they only append to **Log**. Keyen edits the rest.

## Standing instructions

- Consult before structural choices: new scopes, new top-level folders, a new model, dataset or
  variant, what counts as ground truth. Bring a recommended default.
- The primary task is the paper replication plus simple patches. Avoid deviating; contain scope to
  reproducing the tables.
- The released code stays as shipped. Fixes are opt-in variants with their own labels.
- Never store the paper's numbers. Table schema only.
- Don't run ahead: one gate at a time (one fold, then more).
- Side experiments run on 3 folds, not 10.
- When I'm present, one decision at a time. When I'm away, group the decisions with a default for
  each and keep working on the defaults where the step can be undone.
- Queued work counts as approved. If a PC session drops, reconnect and resume without asking.
  Never let the PC sit idle waiting on me for a queued job.
- Run short jobs whenever the PC is idle; only long multi-fold runs wait for night.
- Keep agent cost down: start-up reading is `CLAUDE.md` + `keyen.md` + own handoff only; search
  the logbook. No subagents, workflows or deep research unless I ask.
- Models: pipeline engineer Opus / high; office manager, wiki agent, researcher Sonnet / medium;
  prose passes Fable.
- Track every phase-layer change as a named version.
- Don't read my other folders (thesis, GraphASR); I upload what's needed.

## Open questions

(none yet)

## Decisions made

| date | decision |
|---|---|
| 2026-11-28 | Adopt the graph-asr workflow here: four scopes (office manager, pipeline engineer, researcher, wiki agent), the MkDocs wiki now, paperwork set LOGBOOK.md, keyen.md, docs/sessions/. No manuscript scope until there is a paper tree. |
| 2026-11-28 | Side experiments on 3 folds only. Peephole, Phased, Associative and complex LSTM queues flushed; don't run until I ask. |
| 2026-11-28 | Parked plans (don't start unless asked): complex gated RNN (Wolter & Yao 2018), uRNN/EUNN, xLSTM, Mogrifier LSTM, CIFG / minimal gated unit. |
| 2026-11-28 | Keep the source-accuracy plan and pretrained-model scout threads open. Scout pick: CBraMod, then REVE, LaBraM. |
| 2026-11-27 | All phase-aware work paused; source and target baseline replication takes precedence. |
| 2026-11-27 | Gradient bug: Fix A (`gradCap`, double-precision cap) is the chosen fix (PR 7). |
| 2026-11-27 | Paper's dual-channel transfer row = no transfer; note it in the tables and add a real row from dual-channel source nets on both branches (variant `dualSourceTransfer`). |
| 2026-11-27 | Two verification branches (HPC nets and our threefry nets); one gate at a time. |
| 2026-11-27 | COG-BCI dataset is not introduced now; don't touch it. |
| 2026-11-26 | Queued work counts as approved; resume a dropped PC session without asking. |
| 2026-11-26 | If fast and original layers agree within ~1 point on fold 2, use the fused (fast) layer for phase-aware runs; flag a bigger gap first. |
| 2026-11-25 | The 9 PM rule was only for the old multi-hour layer files; run whenever the PC is idle. |
| 2026-11-25 | Phase-aware training uses the tuned options. Keep the forget bias as is. |
| 2026-11-23 | Keep the unchanged, non-freezing DeepRNN code as the main replication; `freezeIntended` is opt-in. |
| 2026-11-23 | Everything automated and seeded, tables included. Paper numbers never stored. |

## Scratch

## Log

- 2026-11-28: file created from project memory when the workflow was adopted.
