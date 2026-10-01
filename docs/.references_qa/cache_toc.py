from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from copy import deepcopy
from collections import defaultdict, deque

BASE = Path(__file__).parent
OUT = BASE.parent / '项目说明书_双系统验证更新版_参考修订版.docx'
CACHE = BASE / 'fields_refreshed.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}; q = lambda s: '{' + W + '}' + s
txt = lambda p: ''.join(p.xpath('.//w:t/text()', namespaces=NS))
ser = lambda e: E.tostring(e, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(OUT) as z: parts = {n: z.read(n) for n in z.namelist()}
with ZipFile(CACHE) as z:
    cr = E.fromstring(z.read('word/document.xml')); cs = E.fromstring(z.read('word/styles.xml'))
r = E.fromstring(parts['word/document.xml'])
expr = '//w:sdt[w:sdtPr/w:tag[@w:val="TemplateTOC"]]'
toc = r.xpath(expr, namespaces=NS)[0]; cached = cr.xpath(expr, namespaces=NS)[0]
toc.getparent().replace(toc, deepcopy(cached))
anchors = set(cached.xpath('.//w:hyperlink/@w:anchor', namespaces=NS))
destinations = defaultdict(deque)
for p in r.find(q('body')).findall(q('p')):
    # Remove only old TOC bookmarks, preserving all other content and drawing nodes.
    starts = p.xpath('./w:bookmarkStart[starts-with(@w:name,"_Toc")]', namespaces=NS)
    ids = {m.get(q('id')) for m in starts}
    for m in starts: p.remove(m)
    for m in p.findall(q('bookmarkEnd')):
        if m.get(q('id')) in ids: p.remove(m)
    destinations[txt(p)].append(p)
for p in cr.find(q('body')).findall(q('p')):
    marks = [m for m in p.findall(q('bookmarkStart')) if m.get(q('name')) in anchors]
    if not marks: continue
    assert destinations[txt(p)], txt(p)
    dest = destinations[txt(p)].popleft()
    for m in marks:
        dest.insert(1, deepcopy(m))
        ends = [e for e in p.findall(q('bookmarkEnd')) if e.get(q('id')) == m.get(q('id'))]
        assert len(ends) == 1
        dest.append(deepcopy(ends[0]))
styles = E.fromstring(parts['word/styles.xml']); ids = {s.get(q('styleId')) for s in styles.findall(q('style'))}
needed = set(cached.xpath('.//w:pStyle/@w:val|.//w:rStyle/@w:val', namespaces=NS))
for s in cs.findall(q('style')):
    if s.get(q('styleId')) in needed and s.get(q('styleId')) not in ids: styles.append(deepcopy(s))
parts['word/document.xml'] = ser(r); parts['word/styles.xml'] = ser(styles)
with ZipFile(OUT, 'w', ZIP_DEFLATED) as z:
    for n, b in parts.items(): z.writestr(n, b)
assert anchors <= set(r.xpath('//w:bookmarkStart/@w:name', namespaces=NS))
print('Updated TOC entries:', len(anchors))
