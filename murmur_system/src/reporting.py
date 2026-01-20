from __future__ import annotations

from typing import Dict

from pathlib import Path


def write_summary(output_path: Path, sections: Dict[str, str]) -> None:
    lines = []
    lines.append("Murmur Detection System Summary")
    lines.append("=" * 36)
    for title, content in sections.items():
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))
        lines.append(content.strip())
    output_path.write_text("\n".join(lines))
