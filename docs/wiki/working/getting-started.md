# Getting started

1. Clone, then turn on the hooks once:
   `git config core.hooksPath .githooks` (and `git config core.longpaths true` on Windows).
2. Read [`CLAUDE.md`](../../../CLAUDE.md) and [`keyen.md`](../../../keyen.md). Agents also read
   their own handoff in [`docs/sessions/`](../../sessions/README.md).
3. Branch before any work. The rules are in [the git protocol](../../GIT_PROTOCOL.md).
4. To run anything, you need MATLAB with the Deep Learning Toolbox and the full fold files (they are
   not in the repository). Start from [Reproducing a result](reproducing.md).
