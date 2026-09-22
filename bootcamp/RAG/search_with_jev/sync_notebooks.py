"""Convert tutorial Markdown while preserving prose newlines and saved outputs."""

import argparse
import re
import subprocess
from pathlib import Path

import nbformat


def sections(text):
    """Split Python fences from prose without changing Markdown whitespace."""
    offset = 0
    for match in re.finditer(r"^```python\n(.*?)^```[ \t]*$", text, re.M | re.S):
        prose = text[offset : match.start()].strip("\n")
        if prose:
            yield "markdown", prose
        yield "code", match.group(1).rstrip("\n")
        offset = match.end()
    prose = text[offset:].strip("\n")
    if prose:
        yield "markdown", prose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for source in sorted(Path(__file__).parent.glob("*.md")):
        if source.name == "README.md":
            continue
        target = source.with_suffix(".ipynb")
        expected = list(sections(source.read_text()))
        previous = nbformat.read(target, as_version=4) if target.exists() else None
        if args.check:
            assert previous is not None, f"Missing notebook: {target.name}"
            actual = [(c.cell_type, c.source.strip("\n")) for c in previous.cells]
            assert actual == expected, f"Notebook differs from Markdown: {target.name}"
            nbformat.validate(previous)
            print(f"Verified {target.name}")
            continue
        subprocess.run(["uvx", "jupyter-switch", str(source)], check=True)
        notebook = nbformat.read(target, as_version=4)
        # Some converter releases concatenate prose lines. Restore from Markdown.
        notebook.cells = [
            nbformat.v4.new_markdown_cell(text)
            if kind == "markdown"
            else nbformat.v4.new_code_cell(text)
            for kind, text in expected
        ]
        old_code = [c for c in previous.cells if c.cell_type == "code"] if previous else []
        new_code = [c for c in notebook.cells if c.cell_type == "code"]
        if [c.source.rstrip("\n") for c in old_code] == [c.source for c in new_code]:
            for old, new in zip(old_code, new_code):
                new.outputs = old.outputs
                new.execution_count = old.execution_count
        if previous:
            notebook.metadata = previous.metadata
        nbformat.validate(notebook)
        nbformat.write(notebook, target)
        print(f"Synced {target.name}")


if __name__ == "__main__":
    main()
