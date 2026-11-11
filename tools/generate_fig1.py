import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
html_path = REPO_ROOT / "paper" / "figures" / "bifocalgemma_transformer_diagram.html"
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

svg_match = re.search(r"(<svg[\s\S]*?</svg>)", html)
if not svg_match:
    raise ValueError("SVG not found in HTML!")

svg_content = svg_match.group(1)
# Add explicit width, height and white background rect
svg_content = re.sub(
    r'<svg viewBox="0 0 1050 930"[^>]*>',
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1050 930" width="1050" height="930" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif;">\n  <rect width="1050" height="930" fill="#ffffff"/>',
    svg_content,
)

svg_out = REPO_ROOT / "paper" / "figures" / "fig1_transformer_architecture.svg"
with open(svg_out, "w", encoding="utf-8") as f:
    f.write(svg_content)

print("Saved SVG:", svg_out)

# Compile to vector PDF using Typst
typ_path = REPO_ROOT / "paper" / "figures" / "fig1.typ"
pdf_out = REPO_ROOT / "paper" / "figures" / "fig1_transformer_architecture.pdf"

with open(typ_path, "w", encoding="utf-8") as f:
    f.write(
        '#set page(width: 1050pt, height: 930pt, margin: 0pt)\n#image("fig1_transformer_architecture.svg", width: 100%, height: 100%)\n'
    )

res = subprocess.run(["typst", "compile", str(typ_path), str(pdf_out)], capture_output=True, text=True)
print("Typst returncode:", res.returncode)
if res.returncode != 0:
    print("Typst stderr:", res.stderr)
else:
    print("PDF generated successfully:", pdf_out)
    if os.path.exists(typ_path):
        os.remove(typ_path)
