from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from pypdf import PdfReader
from collections import Counter
import json, hashlib, re

base = Path(__file__).parent
src = base.parent / '项目说明书_双系统验证更新版_模板样式.docx'
out = base.parent / '项目说明书_双系统验证更新版_参考修订版.docx'
report = json.loads((base / 'edit_report.json').read_text(encoding='utf-8'))
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
q = lambda n: '{' + ns['w'] + '}' + n
txt = lambda n: ''.join(n.xpath('.//w:t/text()', namespaces=ns))
def tree(e): return (e.tag, tuple(sorted(e.attrib.items())), e.text, tuple(tree(c) for c in e))
with ZipFile(src) as a, ZipFile(out) as b:
    original = E.fromstring(a.read('word/document.xml')); final = E.fromstring(b.read('word/document.xml'))
    for expr in ('//m:oMath', '//wp:anchor', '//w:txbxContent'):
        assert [tree(e) for e in original.xpath(expr, namespaces=ns)] == [tree(e) for e in final.xpath(expr, namespaces=ns)], expr
    media = [n for n in a.namelist() if n.startswith('word/media/')]
    assert all(a.read(n) == b.read(n) for n in media)
    changed = [n for n in a.namelist() if a.read(n) != b.read(n)]
    assert set(changed) <= {'word/document.xml', 'word/styles.xml'}
    toc = final.xpath('//w:sdt[w:sdtPr/w:tag[@w:val="TemplateTOC"]]', namespaces=ns)[0]
    assert not any(s in txt(toc) for s in ('本章参考', '本章证据', '本章实现与证据'))
    assert '参考' in txt(toc)
    assert set(toc.xpath('.//w:hyperlink/@w:anchor', namespaces=ns)) <= set(final.xpath('//w:bookmarkStart/@w:name', namespaces=ns))
    nodes = list(final.find(q('body')))
    heading = next(i for i,n in enumerate(nodes) if n.tag == q('p') and txt(n) == '参考')
    refs = [txt(n) for n in nodes[heading+1:] if n.tag == q('p') and txt(n)]
    assert len(refs) == 6
    assert not any(s in ''.join(refs) for s in ('docs/', 'tests/', 'outputs/', 'doc/CPN-HRL.pdf'))
    assert refs == sorted(refs, key=str.casefold)
    for n in nodes[:heading]:
        if n.tag == q('p'):
            value = txt(n)
            assert not any(value.startswith(s) for s in ('本章参考', '本章证据', '本章实现与证据'))
            assert not re.search(r'（(?:Topcuoglu|Schulman|Hochreiter|Veličković|Yue)[^）]*[，,]\s*\d{4}）', value)

def positions(pdf):
    labels = {}
    for i,p in enumerate(PdfReader(pdf).pages, 1):
        lines = {}
        def visitor(s, cm, tm, font, size):
            compact = re.sub(r'\s', '', s)
            x = round(tm[4]*cm[0]+tm[5]*cm[2]+cm[4],3)
            y = round(tm[4]*cm[1]+tm[5]*cm[3]+cm[5],3)
            if compact and x >= 480:
                lines.setdefault(y, []).append((x, compact))
        p.extract_text(visitor_text=visitor)
        for y, fragments in lines.items():
            fragments.sort(key=lambda v: v[0])
            compact = ''.join(s for x,s in fragments)
            if re.fullmatch(r'（2-[1-7]）', compact):
                labels[compact] = (i, fragments[0][0], y)
    return labels
oldpos = positions(base/'source'/(src.stem+'.pdf'))
newpos = positions(base/'final_v2'/(out.stem+'.pdf'))
assert len(oldpos) == len(newpos) == 7, (oldpos, newpos)
assert oldpos == newpos, (oldpos, newpos)
assert hashlib.sha256(src.read_bytes()).hexdigest() == report['source_sha256']
result = {'references': len(refs), 'chapters_end_indexes_removed': 6,
          'native_equations_unchanged': 7, 'equation_textboxes_unchanged': 7,
          'rendered_equation_label_positions_unchanged': newpos, 'source_unchanged': True,
          'media_preserved': len(media), 'changed_parts': changed}
(base/'final_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
