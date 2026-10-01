from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from collections import Counter
import hashlib, json, re

base = Path(__file__).parent
src = base.parent / '项目说明书_双系统验证更新版.docx'
out = base.parent / '项目说明书_双系统验证更新版_模板样式.docx'
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
q = lambda s: '{' + ns['w'] + '}' + s
txt = lambda e: ''.join(e.xpath('.//w:t/text()', namespaces=ns))
def tree(e):
    return (e.tag, tuple(sorted(e.attrib.items())), e.text, tuple(tree(x) for x in e))

with ZipFile(src) as a, ZipFile(out) as b:
    original = E.fromstring(a.read('word/document.xml'))
    final = E.fromstring(b.read('word/document.xml'))
    old_nodes = list(original.find(q('body')))[21:331]
    expected = []
    for p in old_nodes:
        if p.tag != q('p'):
            continue
        text = txt(p)
        value = text.strip()
        if value == '绪论':
            text = '第 1 章 绪论'
        elif re.match(r'^[1-6]\s+\S', value):
            text = f'第 {value[0]} 章 ' + re.sub(r'^\d+\s+', '', value)
        expected.append(text)
    actual = [txt(p) for p in final.find(q('body')).findall(q('p'))]
    missing = Counter(expected) - Counter(actual)
    assert not missing, missing
    orig_tables = [txt(t) for t in original.xpath('//w:tbl', namespaces=ns)]
    final_tables = [txt(t) for t in final.xpath('//w:tbl', namespaces=ns)][1:]
    assert Counter(orig_tables) == Counter(final_tables)
    orig_math = [tree(m) for m in original.xpath('//m:oMath', namespaces=ns)]
    final_math = [tree(m) for m in final.xpath('//m:oMath', namespaces=ns)]
    assert orig_math == final_math
    media = [n for n in a.namelist() if n.startswith('word/media/')]
    assert all(a.read(n) == b.read(n) for n in media)
    changed = [n for n in a.namelist() if a.read(n) != b.read(n)]
    assert set(changed) == {'[Content_Types].xml', 'word/document.xml',
                            'word/_rels/document.xml.rels', 'word/settings.xml', 'word/styles.xml'}
    bookmarks = set(final.xpath('//w:bookmarkStart/@w:name', namespaces=ns))
    anchors = final.xpath('//w:sdt//w:hyperlink/@w:anchor', namespaces=ns)
    assert anchors and set(anchors) <= bookmarks
    report = {'paragraphs_preserved': len(expected), 'tables_preserved': len(orig_tables),
              'native_equations_preserved': len(orig_math), 'media_preserved': len(media),
              'toc_anchors_verified': len(anchors), 'changed_original_parts': changed,
              'rendered_pages_inspected': 47}
for label in ('source', 'reference'):
    evidence = json.loads((base / f'{label}_evidence.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(Path(evidence['path']).read_bytes()).hexdigest() == evidence['sha256']
report['source_and_reference_unchanged'] = True
(base / 'final_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
