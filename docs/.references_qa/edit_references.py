from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from copy import deepcopy
import hashlib, json, re

BASE = Path(__file__).parent
SRC = BASE.parent / '项目说明书_双系统验证更新版_模板样式.docx'
OUT = BASE.parent / '项目说明书_双系统验证更新版_参考修订版.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W, 'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
q = lambda s: '{' + W + '}' + s
def text(n): return ''.join(n.xpath('.//w:t/text()', namespaces=NS))
def shape_tree(n): return (n.tag, tuple(sorted(n.attrib.items())), n.text, tuple(shape_tree(x) for x in n))
def prop(parent, name, **attrs):
    for old in parent.findall(q(name)): parent.remove(old)
    E.SubElement(parent, q(name), {q(k): str(v) for k, v in attrs.items()})
def replace_plain(p, value):
    assert not p.xpath('.//w:drawing|.//w:pict|.//m:oMath', namespaces=NS)
    runs = p.findall(q('r'))
    fmt = deepcopy(runs[0].find(q('rPr'))) if runs and runs[0].find(q('rPr')) is not None else None
    for node in list(p):
        if node.tag in (q('r'), q('hyperlink')): p.remove(node)
    r = E.SubElement(p, q('r'))
    if fmt is not None: r.append(fmt)
    E.SubElement(r, q('t')).text = value
def replace_fragment(p, old, new):
    # Change text only; keep every drawing, alternate representation and anchor untouched.
    nodes = [t for t in p.xpath('./w:r/w:t|./w:hyperlink/w:r/w:t', namespaces=NS)]
    s = ''.join(t.text or '' for t in nodes)
    assert s.count(old) == 1, (old, s)
    start = s.index(old); end = start + len(old); offset = 0; inserted = False
    for t in nodes:
        v = t.text or ''; a = offset; b = a + len(v); offset = b
        if b <= start or a >= end: continue
        before = v[:max(0, start-a)] if a <= start else ''
        after = v[max(0, end-a):] if b >= end else ''
        t.text = before + (new if not inserted else '') + after
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        inserted = True

with ZipFile(SRC) as z: parts = {n: z.read(n) for n in z.namelist()}
original = E.fromstring(parts['word/document.xml'])
root = E.fromstring(parts['word/document.xml']); body = root.find(q('body'))
sections = {'本章参考与证据', '本章参考与实现定位', '本章实现与证据定位', '本章证据索引', '本章证据定位'}
removed = []; collecting = False; bibliography = {}
for n in list(body):
    value = text(n).strip()
    if n.tag == q('sectPr') or (n.tag == q('p') and re.match(r'^第 [1-6] 章 ', value)):
        collecting = False
    if n.tag == q('p') and value in sections: collecting = True
    if collecting:
        if re.match(r'^(Topcuoglu|Schulman|Hochreiter|Veličković|Yue|Arabnejad) ', value):
            value = value.split(' 原文见项目 ')[0]
            bibliography[value.split(' ', 1)[0]] = value
        removed.append({'tag': E.QName(n).localname, 'text': value})
        body.remove(n)
assert len(bibliography) == 6

edits = []
for p in body.findall(q('p')):
    value = text(p)
    before = value
    if value.startswith('HEFT 根据任务优先级'):
        replace_plain(p, 'Topcuoglu 等（2002）提出的 HEFT 根据任务优先级和异构节点上的预计完成时间进行列表调度，是本项目采用的确定性参考方法。它的规则清晰，适合作为统一比较基准。固定排序和局部放置也有局限，难以覆盖任务顺序、通信和资源竞争的所有组合。本项目将 HEFT 用作基线、学习先验和完整候选。')
    elif value.startswith('学习方法尝试从场景'):
        replace_plain(p, '学习方法尝试从场景和调度状态中学习决策偏好。Schulman 等（2017）提出的 PPO 用于策略更新，Hochreiter 和 Schmidhuber（1997）提出的 LSTM 与 Veličković 等（2018）提出的 GAT 分别用于表示序列和图关系。Yue 等（2026）提出的 CPN-HRL 采用高层 LSTM-PPO、低层 GAT-PPO 的分层结构，为本项目提供了参考。原文研究动态到达、优先级和双队列问题；本项目将其适配到离线 DAG、前驱约束和 makespan 目标。')
    elif value.startswith('CPN-HRL 原文研究动态任务'):
        replace_plain(p, 'Yue 等（2026）提出的 CPN-HRL 研究动态任务到达、优先级、截止期和双队列机制，采用高层 LSTM-PPO 与低层 GAT-PPO 分解决策。本项目参考这一分层思路，为离线 DAG 建立前驱约束、插入时间线和 makespan 奖励。原文的动态双队列、多目标收益和实验数字未沿用。')
    elif value.startswith('搜索分支使用 HEFTSafePortfolioPolicy'):
        replace_fragment(p, 'PEFT 相关候选', 'Arabnejad 和 Barbosa（2014）提出的 PEFT 相关候选')
    if 'openKylin 的实际配置与结果见本章证据定位' in text(p):
        replace_fragment(p, 'openKylin 的实际配置与结果见本章证据定位', 'openKylin 的实际配置与结果见第 4 章')
    if text(p) != before: edits.append({'before': before, 'after': text(p)})

# A final unnumbered chapter uses the retained chapter heading style and page break.
heading = E.Element(q('p')); pp = E.SubElement(heading, q('pPr'))
prop(pp, 'pStyle', val='1'); prop(pp, 'pageBreakBefore'); prop(pp, 'keepNext'); prop(pp, 'keepLines')
E.SubElement(E.SubElement(heading, q('r')), q('t')).text = '参考'
body.insert(len(body)-1, heading)
for author in sorted(bibliography, key=str.casefold):
    p = E.Element(q('p')); pp = E.SubElement(p, q('pPr'))
    prop(pp, 'pStyle', val='a'); prop(pp, 'ind', left=420, hanging=420, firstLineChars=0)
    prop(pp, 'spacing', before=0, after=120, line=400, lineRule='exact'); prop(pp, 'keepLines')
    r = E.SubElement(p, q('r')); rp = E.SubElement(r, q('rPr'))
    prop(rp, 'rFonts', ascii='Times New Roman', hAnsi='Times New Roman', eastAsia='宋体')
    prop(rp, 'sz', val=24); prop(rp, 'color', val='000000')
    E.SubElement(r, q('t')).text = bibliography[author]
    body.insert(len(body)-1, p)

for xpath in ('//m:oMath', '//wp:anchor', '//w:txbxContent'):
    assert [shape_tree(n) for n in original.xpath(xpath, namespaces=NS)] == [shape_tree(n) for n in root.xpath(xpath, namespaces=NS)], xpath
parts['word/document.xml'] = E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(OUT, 'w', ZIP_DEFLATED) as z:
    for n, b in parts.items(): z.writestr(n, b)
report = {'source_sha256': hashlib.sha256(SRC.read_bytes()).hexdigest(),
          'output': str(OUT), 'removed_blocks': removed, 'body_citation_edits': edits,
          'bibliography': bibliography, 'equations_and_textbox_anchors_unchanged': True}
(BASE / 'edit_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('Created:', OUT, '; references:', len(bibliography), '; removed blocks:', len(removed))
