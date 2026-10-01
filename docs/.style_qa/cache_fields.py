from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
from copy import deepcopy as cp
from collections import defaultdict,deque
BASE=Path(__file__).parent
OUT=BASE.parent/'项目说明书_双系统验证更新版_模板样式.docx'
CACHE=BASE/'fields_refreshed.docx'
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS={'w':W}
q=lambda s:'{'+W+'}'+s
def xml(b):return E.fromstring(b)
def ser(n):return E.tostring(n,xml_declaration=True,encoding='UTF-8',standalone=True)
def text(p):return ''.join(p.xpath('.//w:t/text()',namespaces=NS))
z=ZipFile(OUT);c=ZipFile(CACHE);parts={n:z.read(n) for n in z.namelist()}
r=xml(parts['word/document.xml']);cr=xml(c.read('word/document.xml'))
toc=r.xpath('//w:sdt[w:sdtPr/w:tag[@w:val="TemplateTOC"]]',namespaces=NS)[0]
cached=cr.xpath('//w:sdt[w:sdtPr/w:tag[@w:val="TemplateTOC"]]',namespaces=NS)[0]
toc.getparent().replace(toc,cp(cached))
# Bring over only the hidden heading bookmarks needed by TOC hyperlinks.
ps=defaultdict(deque)
for p in r.find(q('body')).findall(q('p')):ps[text(p)].append(p)
for p in cr.find(q('body')).findall(q('p')):
    dest=ps[text(p)].popleft() if ps[text(p)] else None
    if dest is None:continue
    marks=p.xpath('./w:bookmarkStart|./w:bookmarkEnd',namespaces=NS)
    for mark in marks:
        dest.append(cp(mark))
styles=xml(parts['word/styles.xml']);cs=xml(c.read('word/styles.xml'))
ids={s.get(q('styleId')) for s in styles.findall(q('style'))}
needed=set(cached.xpath('.//w:pStyle/@w:val|.//w:rStyle/@w:val',namespaces=NS))
for s in cs.findall(q('style')):
    if s.get(q('styleId')) in needed and s.get(q('styleId')) not in ids:styles.append(cp(s))
parts['word/document.xml']=ser(r);parts['word/styles.xml']=ser(styles)
z.close();c.close()
with ZipFile(OUT,'w',ZIP_DEFLATED) as z:
    for n,d in parts.items():z.writestr(n,d)
print('Cached TOC entries',len(cached.xpath('.//w:p',namespaces=NS)),'and heading bookmarks')
