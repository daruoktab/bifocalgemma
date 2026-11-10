import subprocess
import os

svg_code = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 520" width="1000" height="520" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <rect width="1000" height="520" fill="#ffffff"/>

  <!-- Title & Subtitle -->
  <text x="500" y="32" text-anchor="middle" font-size="18" font-weight="bold" fill="#0f172a">Gemma 4 Cross-Layer KV-Sharing Topology in Decoupled Cross-Attention</text>
  <text x="500" y="52" text-anchor="middle" font-size="11.5" fill="#64748b">Only KV-Owner layers compute dedicated Cross-KV states. KV-Shared layers borrow stationary (K_cross, V_cross) from Designated Anchors.</text>

  <!-- Legend Bar (Y=72..100) -->
  <g transform="translate(100, 72)">
    <!-- Legend item 1 -->
    <rect x="0" y="0" width="16" height="16" rx="3" fill="#d1fae5" stroke="#059669" stroke-width="1.5"/>
    <text x="24" y="12" font-size="11" font-weight="600" fill="#065f46">Sliding KV-Owner (Dedicated Cross-KV)</text>

    <!-- Legend item 2 -->
    <rect x="250" y="0" width="16" height="16" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="274" y="12" font-size="11" font-weight="600" fill="#1e40af">Global Full KV-Owner (512d Head Dim)</text>

    <!-- Legend item 3 -->
    <rect x="510" y="0" width="16" height="16" rx="3" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
    <text x="534" y="12" font-size="11" font-weight="bold" fill="#78350f">Anchor Layer (Full-Length Cache Stored)</text>

    <!-- Legend item 4 -->
    <rect x="760" y="0" width="16" height="16" rx="3" fill="#ffe4e6" stroke="#e11d48" stroke-width="1.5" stroke-dasharray="3 2"/>
    <text x="784" y="12" font-size="11" font-weight="600" fill="#9f1239">KV-Shared Layer (Reuses Anchor KV)</text>
  </g>

  <!-- ========================================================================= -->
  <!-- PANEL A: GEMMA 4 E2B DECODER STACK (35 Layers: 15 Owner + 20 Shared)      -->
  <!-- ========================================================================= -->
  <g transform="translate(40, 115)">
    <!-- Outer Card Frame -->
    <rect x="0" y="0" width="920" height="175" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
    <rect x="16" y="-12" width="410" height="24" rx="5" fill="#334155"/>
    <text x="221" y="4" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#ffffff">PANEL A: GEMMA 4 E2B (35 Layers · 15 KV-Owners / 20 Shared)</text>
    
    <text x="440" y="5" font-size="11" font-weight="600" fill="#059669">KV-Cache Memory Reduction: 57.14% (20 / 35 layers reuse Anchor KV)</text>

    <!-- Layer Grid: 2 rows (Row 1: L00..L17, Row 2: L18..L34) -->
    <!-- Grid Row 1: L00 to L17 -->
'''

# Generate E2B grid (35 layers total, 18 per row)
# E2B: Full indices = [4, 9, 14, 19, 24, 29, 34]
# Owners = 0..14 (15 layers)
# Anchors = [13 (sliding), 14 (full)]
# Shared = 15..34 (20 layers)

def get_e2b_layer(i):
    is_full = i in [4, 9, 14, 19, 24, 29, 34]
    is_anchor = i in [13, 14]
    is_shared = i >= 15
    if is_anchor:
        fill, stroke, sw, dash, txtcol = "#fef3c7", "#d97706", "2.2", "none", "#78350f"
        tag = "Anchor"
    elif is_shared:
        fill, stroke, sw, dash, txtcol = "#ffe4e6", "#e11d48", "1.4", "3 2", "#9f1239"
        tag = "Shared"
    elif is_full:
        fill, stroke, sw, dash, txtcol = "#dbeafe", "#2563eb", "1.5", "none", "#1e40af"
        tag = "Full Own"
    else:
        fill, stroke, sw, dash, txtcol = "#d1fae5", "#059669", "1.4", "none", "#065f46"
        tag = "Slide Own"
    return fill, stroke, sw, dash, txtcol, tag

e2b_nodes = []
for i in range(35):
    row = 0 if i < 18 else 1
    col = i if row == 0 else i - 18
    x = 18 + col * 49
    y = 28 + row * 62
    fill, stroke, sw, dash, txtcol, tag = get_e2b_layer(i)
    dash_attr = f'stroke-dasharray="{dash}"' if dash != "none" else ''
    node = f'''    <rect x="{x}" y="{y}" width="43" height="50" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {dash_attr}/>
    <text x="{x+21.5}" y="{y+19}" text-anchor="middle" font-size="11" font-weight="bold" fill="{txtcol}">L{i:02d}</text>
    <text x="{x+21.5}" y="{y+34}" text-anchor="middle" font-size="8.5" font-weight="600" fill="{txtcol}">{tag}</text>'''
    e2b_nodes.append(node)

svg_code += "\n".join(e2b_nodes)

# Reuse arrows for E2B: curved arrows from L13/L14 to Shared block L15..L34
svg_code += '''
    <!-- Reusable Key-Value Anchor Bridge Indicators -->
    <path d="M 685 40 Q 710 18 735 28" fill="none" stroke="#d97706" stroke-width="2" marker-end="url(#arr-kv)"/>
    <text x="755" y="145" font-size="10.5" font-style="italic" fill="#64748b">← L15–L34 borrow Cross-KV states from Anchor L13 (Sliding) &amp; L14 (Global Full)</text>
  </g>
'''

# Panel B: Gemma 4 E4B Decoder Stack (42 Layers: 24 Owner + 18 Shared)
svg_code += '''
  <!-- ========================================================================= -->
  <!-- PANEL B: GEMMA 4 E4B DECODER STACK (42 Layers · 24 KV-Owners / 18 Shared) -->
  <!-- ========================================================================= -->
  <g transform="translate(40, 310)">
    <!-- Outer Card Frame -->
    <rect x="0" y="0" width="920" height="175" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
    <rect x="16" y="-12" width="410" height="24" rx="5" fill="#334155"/>
    <text x="221" y="4" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#ffffff">PANEL B: GEMMA 4 E4B (42 Layers · 24 KV-Owners / 18 Shared)</text>
    
    <text x="440" y="5" font-size="11" font-weight="600" fill="#059669">KV-Cache Memory Reduction: 42.86% (18 / 42 layers reuse Anchor KV)</text>
'''

def get_e4b_layer(i):
    is_full = i in [5, 11, 17, 23, 29, 35, 41]
    is_anchor = i in [22, 23]
    is_shared = i >= 24
    if is_anchor:
        fill, stroke, sw, dash, txtcol = "#fef3c7", "#d97706", "2.2", "none", "#78350f"
        tag = "Anchor"
    elif is_shared:
        fill, stroke, sw, dash, txtcol = "#ffe4e6", "#e11d48", "1.4", "3 2", "#9f1239"
        tag = "Shared"
    elif is_full:
        fill, stroke, sw, dash, txtcol = "#dbeafe", "#2563eb", "1.5", "none", "#1e40af"
        tag = "Full Own"
    else:
        fill, stroke, sw, dash, txtcol = "#d1fae5", "#059669", "1.4", "none", "#065f46"
        tag = "Slide Own"
    return fill, stroke, sw, dash, txtcol, tag

e4b_nodes = []
for i in range(42):
    row = 0 if i < 21 else 1
    col = i if row == 0 else i - 21
    x = 18 + col * 42
    y = 28 + row * 62
    fill, stroke, sw, dash, txtcol, tag = get_e4b_layer(i)
    dash_attr = f'stroke-dasharray="{dash}"' if dash != "none" else ''
    node = f'''    <rect x="{x}" y="{y}" width="37" height="50" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {dash_attr}/>
    <text x="{x+18.5}" y="{y+19}" text-anchor="middle" font-size="10.5" font-weight="bold" fill="{txtcol}">L{i:02d}</text>
    <text x="{x+18.5}" y="{y+34}" text-anchor="middle" font-size="8" font-weight="600" fill="{txtcol}">{tag}</text>'''
    e4b_nodes.append(node)

svg_code += "\n".join(e4b_nodes)
svg_code += '''
    <text x="440" y="145" font-size="10.5" font-style="italic" fill="#64748b">← L24–L41 borrow Cross-KV states from Anchor L22 (Sliding) &amp; L23 (Global Full)</text>
  </g>
</svg>
'''

svg_path = r"d:\Codings\bifocalgemma\paper\figures\fig2_layer_topology.svg"
pdf_path = r"d:\Codings\bifocalgemma\paper\figures\fig2_layer_topology.pdf"
typ_path = r"d:\Codings\bifocalgemma\paper\figures\fig2.typ"

with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg_code)
print("Saved:", svg_path)

with open(typ_path, "w", encoding="utf-8") as f:
    f.write('#set page(width: 1000pt, height: 520pt, margin: 0pt)\n#image("fig2_layer_topology.svg", width: 100%, height: 100%)\n')

res = subprocess.run(["typst", "compile", typ_path, pdf_path], capture_output=True, text=True)
print("Typst returncode:", res.returncode)
if res.returncode == 0:
    print("PDF generated successfully:", pdf_path)
    if os.path.exists(typ_path):
        os.remove(typ_path)
else:
    print("Typst stderr:", res.stderr)
