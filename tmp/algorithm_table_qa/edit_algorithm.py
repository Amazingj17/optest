from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E

BASE = Path(r'D:\programing\python\optest')
SOURCE = BASE / 'docs/项目说明书.docx'
OUTPUT = BASE / 'docs/项目说明书_算法三线表版.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}
def tag(s): return f'{{{W}}}{s}'
def el(s, **attrs):
    node = E.Element(tag(s))
    for k, v in attrs.items(): node.set(tag(k), str(v))
    return node

with ZipFile(SOURCE) as zin:
    root = E.fromstring(zin.read('word/document.xml'))
    body = root.find('w:body', NS)
    algorithm = next(p for p in body if '清空旧候选' in ''.join(p.xpath('.//w:t/text()', namespaces=NS)))
    caption = algorithm.getprevious()
    assert ''.join(caption.xpath('.//w:t/text()', namespaces=NS)) == '算法2-1 独立组合的核心过程'
    original_text = ''.join(root.xpath('.//w:t/text()', namespaces=NS))

    table = el('tbl')
    props = el('tblPr')
    props.append(el('tblW', w=9404, type='dxa'))
    props.append(el('jc', val='center'))
    props.append(el('tblLayout', type='fixed'))
    borders = el('tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        borders.append(el(edge, val='single' if edge in ('top', 'bottom') else 'nil',
                          sz=12 if edge in ('top', 'bottom') else 0, color='000000'))
    props.append(borders)
    margins = el('tblCellMar')
    for edge in ('top', 'left', 'bottom', 'right'):
        margins.append(el(edge, w=100 if edge in ('top', 'bottom') else 120, type='dxa'))
    props.append(margins)
    table.append(props)
    grid = el('tblGrid'); grid.append(el('gridCol', w=9404)); table.append(grid)

    def cell(header=False):
        row = el('tr'); rpr = el('trPr'); rpr.append(el('cantSplit'))
        if header: rpr.append(el('tblHeader'))
        row.append(rpr)
        tc = el('tc'); tcpr = el('tcPr'); tcpr.append(el('tcW', w=9404, type='dxa'))
        tcpr.append(el('vAlign', val='center'))
        if header:
            b = el('tcBorders'); b.append(el('bottom', val='single', sz=6, color='000000')); tcpr.append(b)
        tc.append(tcpr); row.append(tc); table.append(row)
        return tc

    header = cell(True)
    new_caption = deepcopy(caption)
    ppr = new_caption.find('w:pPr', NS)
    spacing = ppr.find('w:spacing', NS)
    spacing.set(tag('before'), '20'); spacing.set(tag('after'), '20')
    for run in new_caption.findall('w:r', NS):
        rp = run.find('w:rPr', NS)
        if rp is None: rp = el('rPr'); run.insert(0, rp)
        rp.append(el('b')); rp.append(el('color', val='000000'))
    header.append(new_caption)
    content = cell()
    paragraphs = []
    current = el('p')
    for run in algorithm.findall('w:r', NS):
        segment = el('r')
        rp = run.find('w:rPr', NS)
        if rp is not None: segment.append(deepcopy(rp))
        for child in run:
            if child.tag == tag('rPr'): continue
            if child.tag == tag('br'):
                if len(segment) > (1 if rp is not None else 0): current.append(segment)
                paragraphs.append(current); current = el('p')
                segment = el('r')
                if rp is not None: segment.append(deepcopy(rp))
            else: segment.append(deepcopy(child))
        if len(segment) > (1 if rp is not None else 0): current.append(segment)
    paragraphs.append(current)
    assert len(paragraphs) == 10
    for i, p in enumerate(paragraphs):
        pp = el('pPr'); pp.append(el('keepLines'))
        if i < len(paragraphs)-1: pp.append(el('keepNext'))
        pp.append(el('spacing', before=0, after=0, line=300, lineRule='auto'))
        pp.append(el('ind', firstLine=0, firstLineChars=0, left=0))
        pp.append(el('jc', val='left')); pp.append(el('snapToGrid', val=0))
        p.insert(0, pp)
        content.append(p)
    index = body.index(caption)
    body.remove(caption); body.remove(algorithm); body.insert(index, table)
    assert ''.join(root.xpath('.//w:t/text()', namespaces=NS)) == original_text
    xml = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
    with ZipFile(OUTPUT, 'w') as zout:
        for info in zin.infolist():
            zout.writestr(info, xml if info.filename == 'word/document.xml' else zin.read(info.filename))

with ZipFile(SOURCE) as a, ZipFile(OUTPUT) as b:
    changes = [n for n in a.namelist() if a.read(n) != b.read(n)]
    assert changes == ['word/document.xml'], changes
print('Edited algorithm into a three-line table; exact text and all other package parts preserved.')
print(OUTPUT)
