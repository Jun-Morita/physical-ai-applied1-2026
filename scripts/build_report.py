"""Print the report (docs/report/report.html) to PDF with headless Chrome.

    python3 scripts/build_report.py \\
      --chrome "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" \\
      [--youtube https://youtu.be/...]

The HTML is the report source; each <section class="page"> is one A4 page.
Windows Chrome works from WSL (paths are converted with wslpath).
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/report"
PDF_NAME = "説明資料_PAI最終課題_Morita_Junichiro.pdf"


def windows_path(path):
    return subprocess.run(
        ["wslpath", "-w", str(path)], capture_output=True, text=True, check=True
    ).stdout.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chrome", type=Path, required=True)
    p.add_argument("--youtube", help="Video URL shown under the title")
    a = p.parse_args()
    html = (REPORT / "report.html").read_text(encoding="utf-8")
    if a.youtube:
        html = re.sub(
            r"(<span data-youtube>)[^<]*(</span>)", rf"\g<1>{a.youtube}\g<2>", html
        )
    # Print a copy next to the figures so relative image paths resolve.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", dir=REPORT, encoding="utf-8", delete=False
    ) as page:
        page.write(html)
    try:
        pdf = REPORT / PDF_NAME
        subprocess.run(
            [
                str(a.chrome),
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={windows_path(pdf)}",
                "file:///" + windows_path(page.name).replace("\\", "/"),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        Path(page.name).unlink()
    print("saved", pdf)


if __name__ == "__main__":
    main()
