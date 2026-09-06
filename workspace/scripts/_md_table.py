"""Minimal GFM table writer for workspace/results/*.md."""

from __future__ import annotations

from pathlib import Path

MEASURED_HEADING = "本仓库实测"
MEASURED_START = "<!-- MEASURED:START -->"
MEASURED_END = "<!-- MEASURED:END -->"


def _cell(value) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_md_table(headers: list[str], rows: list[dict]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def write_md_table(
    path: Path,
    headers: list[str],
    rows: list[dict],
    title: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    if title:
        parts.extend([f"# {title}", ""])
    parts.append(render_md_table(headers, rows))
    parts.append("")
    path.write_text("\n".join(parts), encoding="utf-8")


def replace_measured_section(
    path: Path,
    headers: list[str],
    rows: list[dict],
    heading: str = MEASURED_HEADING,
) -> None:
    """Replace the measured block; keep 原论文结果 and other prose."""
    table = render_md_table(headers, rows)
    block = (
        f"## {heading}\n\n"
        f"{MEASURED_START}\n"
        f"{table}\n"
        f"{MEASURED_END}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(block, encoding="utf-8")
        return
    text = path.read_text(encoding="utf-8")
    if MEASURED_START in text and MEASURED_END in text:
        start = text.index(MEASURED_START)
        end = text.index(MEASURED_END) + len(MEASURED_END)
        heading_at = text.rfind(f"## {heading}", 0, start)
        prefix = text[:heading_at] if heading_at != -1 else text[:start]
        suffix = text[end:].lstrip("\n")
        path.write_text(prefix + block + (("\n" + suffix) if suffix else ""), encoding="utf-8")
        return
    heading_at = text.find(f"## {heading}")
    if heading_at != -1:
        path.write_text(text[:heading_at] + block, encoding="utf-8")
        return
    path.write_text(text.rstrip() + "\n\n" + block, encoding="utf-8")
