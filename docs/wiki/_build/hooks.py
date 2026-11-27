"""MkDocs build hooks for the Phase-Aware LSTM wiki.

Nothing is written into the tree. At build time this module:

1. mirrors every tracked .md outside docs/wiki/ as a page under repo/<path>, so READMEs,
   result notes and the contract are searchable in-site;
2. generates four reference pages: tree map, tables and notes, results index, decisions index;
3. resolves every relative link against the page's own source file: a link to a rendered page
   stays in-site, anything else goes to GitHub (or to the local file with WIKI_LINKS=local).

Settings for this repository are the constants below.
"""

from __future__ import annotations

import csv
import io
import os
import posixpath
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from mkdocs.structure.files import File

REPO_URL = "https://github.com/backslash-keyen/Phase-Aware-LSTM"
BRANCH = "main"
WIKI_DIR = "docs/wiki"
MIRROR_PREFIX = "repo"
# Tracked paths that are never mirrored (the released code's MATLAB project metadata).
MIRROR_SKIP = ("resources/",)
# Folders shown as one line with a file count in the tree map.
TREE_COLLAPSE = (
    "resources/project/",
    "replication/versions/fast/",
    "replication/versions/originals/",
    "replication/variants/",
    "utils/",
)
# Where result notes and generated tables live in the tree.
RESULT_NOTES = ("replication/results/",)
TABLE_EXT = (".csv", ".tex")
# Untracked run outputs on the PC (the folder holding manifest.csv), if set.
RESULTS_ENV = "PLSTM_RESULTS"
DECISIONS_FILE = "keyen.md"
DECISIONS_HEADING = "Decisions made"

ROOT = Path(__file__).resolve().parents[3]
LINK_RE = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")

_tracked: list[str] = []
_mirrored: set[str] = set()


def _git_files() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.splitlines() if p]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")


# ---------------------------------------------------------------- generated pages

def _tree_page() -> str:
    lines = [
        "# Tree map",
        "",
        "Every tracked file, by folder, computed from `git ls-files` at build time. Large folders "
        "are shown as one line with a file count.",
        "",
    ]
    collapsed = defaultdict(int)
    by_dir = defaultdict(list)
    for p in _tracked:
        hit = next((c for c in TREE_COLLAPSE if p.startswith(c)), None)
        if hit:
            collapsed[hit] += 1
        else:
            by_dir[posixpath.dirname(p) or "."].append(posixpath.basename(p))
    lines += ["| folder | files |", "|---|---|"]
    for c in sorted(collapsed):
        lines.append(f"| [`{c}`]({c}) | {collapsed[c]} files |")
    lines += ["", ""]
    for d in sorted(by_dir):
        lines.append(f"## `{d}/`" if d != "." else "## repository root")
        lines.append("")
        for name in sorted(by_dir[d], key=str.lower):
            path = name if d == "." else f"{d}/{name}"
            lines.append(f"- [`{name}`]({path})")
        lines.append("")
    return "\n".join(lines)


def _tables_page() -> str:
    notes = [p for p in _tracked if p.startswith(RESULT_NOTES) and p.endswith(".md")]
    tables = [p for p in _tracked if p.endswith(TABLE_EXT) and (p.startswith("docs/") or p.startswith(RESULT_NOTES))]
    lines = [
        "# Tables and notes",
        "",
        "Result notes and table sources tracked in the repository, computed at build time. The paper's "
        "tables are built on the PC by `buildTables` from fresh runs (see the "
        "[replication README](replication/README.md)); the paper's own numbers are never stored.",
        "",
        "## Result notes",
        "",
    ]
    lines += [f"- [`{p}`]({p})" for p in sorted(notes)] or ["Not written yet."]
    lines += ["", "## Table sources", ""]
    lines += [f"- [`{p}`]({p})" for p in sorted(tables)] or ["None tracked yet."]
    return "\n".join(lines) + "\n"


def _results_page() -> str:
    lines = [
        "# Results index",
        "",
        "Experiments logged in the replication manifest. The manifest is written on the PC by "
        "`runReplication` and is not tracked; set `" + RESULTS_ENV + "` to the folder holding "
        "`manifest.csv` (normally `replication/results` in the PC clone) to list it here.",
        "",
    ]
    candidates = []
    if os.environ.get(RESULTS_ENV):
        candidates.append(Path(os.environ[RESULTS_ENV]) / "manifest.csv")
    candidates.append(ROOT / "replication" / "results" / "manifest.csv")
    manifest = next((c for c in candidates if c.is_file()), None)
    if manifest is None:
        lines.append("Not run yet: no `manifest.csv` found on this machine.")
        return "\n".join(lines) + "\n"
    rows = list(csv.DictReader(io.StringIO(manifest.read_text(encoding="utf-8", errors="replace"))))
    lines.append(f"From `{manifest}`: {len(rows)} rows.")
    lines.append("")
    keys = [k for k in ("id", "status", "start", "end", "resultFile") if rows and k in rows[0]]
    if not keys and rows:
        keys = list(rows[0].keys())[:5]
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "---|" * len(keys))
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")).replace("|", "/") for k in keys) + " |")
    return "\n".join(lines) + "\n"


def _decisions_page() -> str:
    lines = [
        "# Decisions index",
        "",
        f"Parsed from the *{DECISIONS_HEADING}* table in [`{DECISIONS_FILE}`]({DECISIONS_FILE}), "
        "newest first. That file is the authority.",
        "",
    ]
    text = _read(DECISIONS_FILE) if (ROOT / DECISIONS_FILE).is_file() else ""
    m = re.search(rf"^## {re.escape(DECISIONS_HEADING)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    rows = []
    if m:
        for line in m.group(1).splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and re.match(r"\d{4}-\d{2}-\d{2}$", cells[0]):
                rows.append((cells[0], "|".join(cells[1:]).strip()))
    if not rows:
        lines.append("No decisions recorded yet.")
    else:
        rows.sort(key=lambda r: r[0], reverse=True)
        lines += ["| date | decision |", "|---|---|"]
        lines += [f"| {d} | {t} |" for d, t in rows]
    return "\n".join(lines) + "\n"


GENERATED = {
    "reference/tree.md": _tree_page,
    "reference/tables.md": _tables_page,
    "reference/results.md": _results_page,
    "reference/decisions.md": _decisions_page,
}


# ---------------------------------------------------------------- mkdocs events

def on_files(files, config):
    global _tracked
    _tracked = _git_files()
    _mirrored.clear()
    wiki_prefix = WIKI_DIR + "/"
    for path in _tracked:
        if not path.endswith(".md") or path.startswith(wiki_prefix) or path.startswith(MIRROR_SKIP):
            continue
        uri = f"{MIRROR_PREFIX}/{path}"
        files.append(File.generated(config, uri, content=_read(path)))
        _mirrored.add(path)
    for uri, build in GENERATED.items():
        files.append(File.generated(config, uri, content=build()))
    return files


def _real_path(src_uri: str) -> str:
    """Path in the repository that a page's links are relative to."""
    if src_uri.startswith(MIRROR_PREFIX + "/"):
        return src_uri[len(MIRROR_PREFIX) + 1:]
    if src_uri in GENERATED:
        # Generated pages write repo-root-relative links.
        return "_root_page.md"
    return f"{WIKI_DIR}/{src_uri}"


def _outside_url(target: str) -> str:
    if os.environ.get("WIKI_LINKS") == "local":
        return (ROOT / target).resolve().as_uri()
    kind = "tree" if (ROOT / target).is_dir() else "blob"
    return f"{REPO_URL}/{kind}/{BRANCH}/{target}".rstrip("/")


def _rewrite(link: str, page_src: str) -> str:
    if re.match(r"^[a-z][a-z0-9+.-]*:", link, re.I) or link.startswith(("#", "/")):
        return link
    path, _, anchor = link.partition("#")
    if not path:
        return link
    base = posixpath.dirname(_real_path(page_src))
    target = posixpath.normpath(posixpath.join(base, path))
    if target.startswith(".."):
        return link
    if target.startswith(WIKI_DIR + "/"):
        site_src = target[len(WIKI_DIR) + 1:]
    elif target in _mirrored:
        site_src = f"{MIRROR_PREFIX}/{target}"
    else:
        url = _outside_url(target)
        return url + (f"#{anchor}" if anchor else "")
    rel = posixpath.relpath(site_src, posixpath.dirname(page_src) or ".")
    return rel + (f"#{anchor}" if anchor else "")


def on_page_markdown(markdown, page, config, files):
    src = page.file.src_uri
    out, in_fence = [], False
    for line in markdown.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        if not in_fence:
            line = LINK_RE.sub(lambda m: m.group(1) + _rewrite(m.group(2), src) + m.group(3), line)
        out.append(line)
    text = "\n".join(out)
    if src.startswith(MIRROR_PREFIX + "/"):
        real = _real_path(src)
        text = (
            f'!!! note "Mirrored from the repository"\n'
            f"    Rendered from [`{real}`]({_outside_url(real)}) at build time. Edit that file, not this page.\n\n"
            + text
        )
    return text
