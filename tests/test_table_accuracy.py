"""Regression tests for selecting ground-truth tables from arXiv sources."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from table_accuracy import extract_source_tables, tex_files  # noqa: E402


def _archive(files: dict[str, str]) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for name, text in files.items():
            payload = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return out.getvalue()


def test_only_referenced_tex_files_become_ground_truth():
    blob = _archive({
        "main.tex": (r"\documentclass{article}\input{used}"
                     r"\begin{document}\end{document}"),
        "used.tex": r"\begin{tabular}{lr}A&1\\B&2\\C&3\\D&4\end{tabular}",
        "unused.tex": r"\begin{tabular}{lr}X&5\\Y&6\\Z&7\\W&8\end{tabular}",
    })
    tables = extract_source_tables(blob)
    assert len(tables) == 1
    assert tables[0].numbers == ["1", "2", "3", "4"]


def test_commented_input_is_not_followed():
    blob = _archive({
        "main.tex": ("\\documentclass{article}\n% \\input{unused}\n"
                     r"\begin{tabular}{lr}A&1\\B&2\\C&3\\D&4\end{tabular}"),
        "unused.tex": r"\begin{tabular}{lr}X&5\\Y&6\\Z&7\\W&8\end{tabular}",
    })
    docs = list(tex_files(blob))
    assert len(docs) == 1
    assert "X&5" not in docs[0]
