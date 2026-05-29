from fpdf import FPDF
import re

with open('/Users/admin/Documents/Los-Altos-Hacks/technical_paper.md', 'r') as f:
    content = f.read()

# Normalise special unicode to ASCII-safe equivalents
replacements = {
    '\u00b2': '2', '\u00b3': '3', '\u00b9': '1',
    '\u207a': '+', '\u207b': '-', '\u00b0': 'deg',
    '\u00e9': 'e', '\u00fc': 'u', '\u00f6': 'o', '\u00e4': 'a',
    '\u03ba': 'k', '\u03b5': 'e', '\u03bb': 'l',
    '\u0394': 'D', '\u03b1': 'a', '\u03b2': 'b',
    '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
    '\u2013': '-', '\u2014': '--', '\u2026': '...',
    '\u00d7': 'x', '\u00f7': '/',
    '\u2192': '->', '\u2190': '<-', '\u2248': '~=',
    '\u00b1': '+/-', '\u2265': '>=', '\u2264': '<=',
    '\u00e0': 'a', '\u00e8': 'e', '\u00ec': 'i',
    '\u00c9': 'E', '\u00dc': 'U',
    # Superscripts
    '\u00b2': '2', '\u00b3': '3',
    # Chemical
    '\u2082': '2', '\u2081': '1',
    '\u00c5': 'A',  # Angstrom
}
for k, v in replacements.items():
    content = content.replace(k, v)

# Strip any remaining non-latin1 chars
content = content.encode('latin-1', errors='replace').decode('latin-1')


class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'PETase-ML Technical Paper   |   Page {self.page_no()}', align='C')


def strip_md(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    return text.strip()


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()
pdf.set_margins(22, 22, 22)

lines = content.split('\n')
i = 0
in_code = False

while i < len(lines):
    line = lines[i]

    # Code block toggle
    if line.strip().startswith('```'):
        if not in_code:
            in_code = True
            pdf.ln(2)
            pdf.set_fill_color(245, 245, 248)
        else:
            in_code = False
            pdf.ln(2)
        i += 1
        continue

    if in_code:
        pdf.set_font('Courier', '', 7.5)
        pdf.set_text_color(40, 60, 40)
        pdf.set_fill_color(245, 245, 248)
        code_line = line.rstrip()
        if code_line:
            # Truncate very long code lines to avoid layout errors
            if len(code_line) > 100:
                code_line = code_line[:97] + '...'
            pdf.set_x(22)
            pdf.multi_cell(166, 5, code_line, fill=True)
        else:
            pdf.ln(2)
        i += 1
        continue

    # H1 — title
    if re.match(r'^# [^#]', line):
        pdf.set_font('Helvetica', 'B', 17)
        pdf.set_text_color(10, 60, 140)
        text = line[2:].strip()
        pdf.multi_cell(0, 10, text, align='C')
        pdf.ln(5)

    # H2
    elif re.match(r'^## [^#]', line):
        pdf.ln(4)
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(10, 60, 140)
        text = line[3:].strip()
        pdf.multi_cell(0, 8, text)
        pdf.set_draw_color(10, 60, 140)
        pdf.set_line_width(0.5)
        pdf.line(22, pdf.get_y(), 188, pdf.get_y())
        pdf.ln(3)

    # H3
    elif re.match(r'^### [^#]', line):
        pdf.ln(3)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(30, 30, 30)
        text = line[4:].strip()
        pdf.multi_cell(0, 7, text)
        pdf.ln(1)

    # Horizontal rule
    elif line.strip() == '---':
        pdf.ln(2)
        pdf.set_draw_color(200, 200, 200)
        pdf.set_line_width(0.3)
        pdf.line(22, pdf.get_y(), 188, pdf.get_y())
        pdf.ln(3)

    # Table row
    elif line.strip().startswith('|'):
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        # Separator row
        if all(re.match(r'^[-: ]+$', c) for c in cells if c):
            i += 1
            continue
        n = len(cells)
        col_w = 166 / max(n, 1)
        # Detect header: next non-empty line is separator
        is_header = False
        if i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if nxt.startswith('|') and all(re.match(r'^[-: ]+$', c) for c in nxt.strip('|').split('|') if c):
                is_header = True
        if is_header:
            pdf.set_font('Helvetica', 'B', 8)
            pdf.set_fill_color(210, 228, 255)
            pdf.set_text_color(10, 30, 90)
        else:
            pdf.set_font('Helvetica', '', 8)
            pdf.set_fill_color(248, 250, 255)
            pdf.set_text_color(30, 30, 30)
        pdf.set_draw_color(180, 200, 230)
        pdf.set_line_width(0.2)
        for ci, cell in enumerate(cells):
            w = col_w * 2 if (n == 2 and ci == 1) else col_w
            pdf.cell(col_w, 6, strip_md(cell)[:80], border=1, fill=True)
        pdf.ln()

    # Bullet
    elif re.match(r'^[\-\*] ', line.strip()):
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(30, 30, 30)
        text = strip_md(line.strip()[2:])
        x = pdf.get_x()
        pdf.set_x(27)
        pdf.cell(5, 6, chr(149))
        pdf.multi_cell(0, 6, text)

    # Numbered list
    elif re.match(r'^\d+\. ', line.strip()):
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(30, 30, 30)
        m = re.match(r'^(\d+)\. (.*)', line.strip())
        if m:
            pdf.set_x(27)
            pdf.cell(7, 6, m.group(1) + '.')
            pdf.multi_cell(0, 6, strip_md(m.group(2)))

    # **Abstract** bold label paragraph
    elif line.strip().startswith('**') and '**' in line.strip()[2:]:
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(50, 50, 50)
        text = strip_md(line.strip())
        pdf.multi_cell(0, 6, text)

    # Normal text
    elif line.strip():
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(30, 30, 30)
        text = strip_md(line.strip())
        pdf.multi_cell(0, 6, text)
        pdf.ln(1)

    else:
        pdf.ln(2)

    i += 1

out = '/Users/admin/Documents/Los-Altos-Hacks/PETase_ML_Technical_Paper.pdf'
pdf.output(out)
print(f'Done: {out}')
