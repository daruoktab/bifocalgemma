import os
import subprocess

svg_code = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1060 620" width="1060" height="620" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <rect width="1060" height="620" fill="#ffffff"/>

  <!-- Title & Subtitle -->
  <text x="530" y="30" text-anchor="middle" font-size="17" font-weight="bold" fill="#0f172a">Gemma 4 Cross-Layer KV-Sharing Topology in Decoupled Cross-Attention</text>
  <text x="530" y="49" text-anchor="middle" font-size="11" fill="#475569">Dedicated BF16 Cross-KV projections exist strictly on KV-Owner layers. KV-Shared layers reuse stationary (K_cross, V_cross) states from designated Anchor layers.</text>

  <!-- Legend Bar (Y=64..88) -->
  <g transform="translate(70, 64)">
    <rect x="0" y="0" width="16" height="16" rx="3" fill="#ecfdf5" stroke="#059669" stroke-width="1.4"/>
    <text x="22" y="12" font-size="11" font-weight="600" fill="#065f46">Sliding KV-Owner (Dedicated Cross-KV)</text>

    <rect x="255" y="0" width="16" height="16" rx="3" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
    <text x="277" y="12" font-size="11" font-weight="600" fill="#1e40af">Global Full KV-Owner (512d Head Dim)</text>

    <rect x="525" y="0" width="16" height="16" rx="3" fill="#fef3c7" stroke="#d97706" stroke-width="2.2"/>
    <text x="547" y="12" font-size="11" font-weight="bold" fill="#78350f">Anchor Layer (Cached Cross-KV Provider)</text>

    <rect x="800" y="0" width="16" height="16" rx="3" fill="#fff1f2" stroke="#e11d48" stroke-width="1.4" stroke-dasharray="3 2"/>
    <text x="822" y="12" font-size="11" font-weight="600" fill="#9f1239">KV-Shared Layer (Reuses Anchor Cache)</text>
  </g>

  <!-- ========================================================================= -->
  <!-- PANEL A: GEMMA 4 E2B DECODER STACK (35 Layers: 15 Owner + 20 Shared)      -->
  <!-- ========================================================================= -->
  <g transform="translate(30, 105)">
    <!-- Outer Card Frame -->
    <rect x="0" y="0" width="1000" height="230" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
    <rect x="18" y="-12" width="420" height="24" rx="5" fill="#1e293b"/>
    <text x="228" y="4" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#ffffff">PANEL A: GEMMA 4 E2B (35 Layers · 15 KV-Owners / 20 Shared)</text>
    
    <rect x="630" y="-12" width="350" height="24" rx="5" fill="#ecfdf5" stroke="#059669" stroke-width="1.2"/>
    <text x="805" y="4" text-anchor="middle" font-size="11" font-weight="bold" fill="#065f46">Cross-KV Cache Reduction: 57.14% (20 / 35 Layers)</text>
'''

# Generate E2B grid (35 layers total: Row 1 = 18 layers, Row 2 = 17 layers)
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
        tag = "Slide Anc" if i == 13 else "Full Anc"
    elif is_shared:
        fill, stroke, sw, dash, txtcol = "#fff1f2", "#e11d48", "1.4", "3 2", "#9f1239"
        tag = "Shared"
    elif is_full:
        fill, stroke, sw, dash, txtcol = "#eff6ff", "#2563eb", "1.5", "none", "#1e40af"
        tag = "Full Own"
    else:
        fill, stroke, sw, dash, txtcol = "#ecfdf5", "#059669", "1.4", "none", "#065f46"
        tag = "Slide Own"
    return fill, stroke, sw, dash, txtcol, tag

e2b_nodes = []
for i in range(35):
    row = 0 if i < 18 else 1
    col = i if row == 0 else i - 18
    x = 23 + col * 53
    y = 26 + row * 52
    fill, stroke, sw, dash, txtcol, tag = get_e2b_layer(i)
    dash_attr = f'stroke-dasharray="{dash}"' if dash != "none" else ''
    node = f'''    <rect x="{x}" y="{y}" width="48" height="46" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {dash_attr}/>
    <text x="{x+24}" y="{y+18}" text-anchor="middle" font-size="11" font-weight="bold" fill="{txtcol}">L{i:02d}</text>
    <text x="{x+24}" y="{y+34}" text-anchor="middle" font-size="8.5" font-weight="600" fill="{txtcol}">{tag}</text>'''
    e2b_nodes.append(node)

svg_code += "\n".join(e2b_nodes)

# Annotation Banner for Panel A
svg_code += '''
    <!-- Annotation & Routing Banner -->
    <rect x="23" y="140" width="954" height="74" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.2"/>
    
    <!-- Left Box: Anchor Badge -->
    <rect x="35" y="148" width="195" height="58" rx="6" fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
    <text x="132" y="168" text-anchor="middle" font-size="11" font-weight="bold" fill="#78350f">★ DESIGNATED ANCHORS</text>
    <text x="132" y="184" text-anchor="middle" font-size="9.5" font-weight="semibold" fill="#92400e">L13: Sliding (w=512)</text>
    <text x="132" y="198" text-anchor="middle" font-size="9.5" font-weight="semibold" fill="#92400e">L14: Global Full (512d)</text>

    <!-- Center Arrow -->
    <text x="245" y="181" font-size="16" font-weight="bold" fill="#d97706">→</text>

    <!-- Right Box: Shared Layer Routing -->
    <rect x="265" y="148" width="700" height="58" rx="6" fill="#fff1f2" stroke="#fecdd3" stroke-width="1.2"/>
    <text x="280" y="169" font-size="11.5" font-weight="bold" fill="#9f1239">20 Downstream Layers (L15–L34) completely bypass Cross-KV Projections</text>
    <text x="280" y="185" font-size="10.5" fill="#475569">Sliding layers reuse L13 Cross-KV; Global layers (L19, L24, L29, L34) reuse L14 Cross-KV. States computed once during prefill.</text>
    <text x="280" y="199" font-size="10" font-weight="600" fill="#059669">Result: 57.14% reduction in cross-attention KV-cache memory footprint with zero quality degradation.</text>
  </g>
'''

# Panel B: Gemma 4 E4B Decoder Stack (42 Layers: 24 Owner + 18 Shared)
svg_code += '''
  <!-- ========================================================================= -->
  <!-- PANEL B: GEMMA 4 E4B DECODER STACK (42 Layers · 24 KV-Owners / 18 Shared) -->
  <!-- ========================================================================= -->
  <g transform="translate(30, 360)">
    <!-- Outer Card Frame -->
    <rect x="0" y="0" width="1000" height="230" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
    <rect x="18" y="-12" width="420" height="24" rx="5" fill="#1e293b"/>
    <text x="228" y="4" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#ffffff">PANEL B: GEMMA 4 E4B (42 Layers · 24 KV-Owners / 18 Shared)</text>
    
    <rect x="630" y="-12" width="350" height="24" rx="5" fill="#ecfdf5" stroke="#059669" stroke-width="1.2"/>
    <text x="805" y="4" text-anchor="middle" font-size="11" font-weight="bold" fill="#065f46">Cross-KV Cache Reduction: 42.86% (18 / 42 Layers)</text>
'''

def get_e4b_layer(i):
    is_full = i in [5, 11, 17, 23, 29, 35, 41]
    is_anchor = i in [22, 23]
    is_shared = i >= 24
    if is_anchor:
        fill, stroke, sw, dash, txtcol = "#fef3c7", "#d97706", "2.2", "none", "#78350f"
        tag = "Slide Anc" if i == 22 else "Full Anc"
    elif is_shared:
        fill, stroke, sw, dash, txtcol = "#fff1f2", "#e11d48", "1.4", "3 2", "#9f1239"
        tag = "Shared"
    elif is_full:
        fill, stroke, sw, dash, txtcol = "#eff6ff", "#2563eb", "1.5", "none", "#1e40af"
        tag = "Full Own"
    else:
        fill, stroke, sw, dash, txtcol = "#ecfdf5", "#059669", "1.4", "none", "#065f46"
        tag = "Slide Own"
    return fill, stroke, sw, dash, txtcol, tag

e4b_nodes = []
for i in range(42):
    row = 0 if i < 21 else 1
    col = i if row == 0 else i - 21
    x = 28 + col * 45
    y = 26 + row * 52
    fill, stroke, sw, dash, txtcol, tag = get_e4b_layer(i)
    dash_attr = f'stroke-dasharray="{dash}"' if dash != "none" else ''
    node = f'''    <rect x="{x}" y="{y}" width="40" height="46" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {dash_attr}/>
    <text x="{x+20}" y="{y+18}" text-anchor="middle" font-size="10.5" font-weight="bold" fill="{txtcol}">L{i:02d}</text>
    <text x="{x+20}" y="{y+34}" text-anchor="middle" font-size="8" font-weight="600" fill="{txtcol}">{tag}</text>'''
    e4b_nodes.append(node)

svg_code += "\n".join(e4b_nodes)

# Annotation Banner for Panel B
svg_code += '''
    <!-- Annotation & Routing Banner -->
    <rect x="23" y="140" width="954" height="74" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.2"/>
    
    <!-- Left Box: Anchor Badge -->
    <rect x="35" y="148" width="195" height="58" rx="6" fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
    <text x="132" y="168" text-anchor="middle" font-size="11" font-weight="bold" fill="#78350f">★ DESIGNATED ANCHORS</text>
    <text x="132" y="184" text-anchor="middle" font-size="9.5" font-weight="semibold" fill="#92400e">L22: Sliding (w=512)</text>
    <text x="132" y="198" text-anchor="middle" font-size="9.5" font-weight="semibold" fill="#92400e">L23: Global Full (512d)</text>

    <!-- Center Arrow -->
    <text x="245" y="181" font-size="16" font-weight="bold" fill="#d97706">→</text>

    <!-- Right Box: Shared Layer Routing -->
    <rect x="265" y="148" width="700" height="58" rx="6" fill="#fff1f2" stroke="#fecdd3" stroke-width="1.2"/>
    <text x="280" y="169" font-size="11.5" font-weight="bold" fill="#9f1239">18 Downstream Layers (L24–L41) completely bypass Cross-KV Projections</text>
    <text x="280" y="185" font-size="10.5" fill="#475569">Sliding layers reuse L22 Cross-KV; Global layers (L29, L35, L41) reuse L23 Cross-KV. States computed once during prefill.</text>
    <text x="280" y="199" font-size="10" font-weight="600" fill="#059669">Result: 42.86% reduction in cross-attention KV-cache memory footprint with zero quality degradation.</text>
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
    f.write('#set page(width: 1060pt, height: 620pt, margin: 0pt)\n#image("fig2_layer_topology.svg", width: 100%, height: 100%)\n')

res = subprocess.run(["typst", "compile", typ_path, pdf_path], capture_output=True, text=True)
print("Typst returncode:", res.returncode)
if res.returncode == 0:
    print("PDF generated successfully:", pdf_path)
    if os.path.exists(typ_path):
        os.remove(typ_path)
else:
    print("Typst stderr:", res.stderr)
