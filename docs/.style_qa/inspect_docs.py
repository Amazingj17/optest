from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from collections import Counter
import json, hashlib

BASE=Path(__file__).parent
REF=Path(r'D:\programing\python\optest归档\决赛作品提交模板\操作系统开源创新大赛项目说明书.docx')
SRC=Path(r'D:\programing\python\optest\docs\项目说明书_双系统验证更新版.docx')
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def txt(e): return ''.join(e.xpath('.//w:t/text()',namespaces=NS))
def clean(e):
    if e is None:return None
    return {E.QName(e).localname: {'attrs':{E.QName(k).localname:v for k,v in e.attrib.items()},'children':[clean(c) for c in e]}}
for label,path in [('reference',REF),('source',SRC)]:
    z=ZipFile(path); root=E.fromstring(z.read('word/document.xml')); styles=E.fromstring(z.read('word/styles.xml'))
    report={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'styles':clean(styles),'sections':[clean(s) for s in root.xpath('//w:sectPr',namespaces=NS)],
      'blocks':[{'index':i,'tag':E.QName(n).localname,'text':txt(n),'pPr':clean(n.find('w:pPr',NS)),
        'runs':[{'text':txt(r),'rPr':clean(r.find('w:rPr',NS))} for r in n.findall('w:r',NS)]} for i,n in enumerate(root.find('w:body',NS))],
      'package':[{'name':n,'size':len(z.read(n)),'sha256':hashlib.sha256(z.read(n)).hexdigest()} for n in z.namelist()],
      'tables':[clean(t.find('w:tblPr',NS)) for t in root.xpath('//w:tbl',namespaces=NS)],
      'headers_footers':{n:clean(E.fromstring(z.read(n))) for n in z.namelist() if n.startswith(('word/header','word/footer')) and n.endswith('.xml')}}
    (BASE/(label+'_evidence.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    outline='\n'.join(f"{b['index']:3} {b['tag']:8} {root.find('w:body',NS)[b['index']].xpath('./w:pPr/w:pStyle/@w:val',namespaces=NS)} {b['text'][:200]}" for b in report['blocks'])
    (BASE/(label+'_outline.txt')).write_text(outline,encoding='utf8')
    print(label,'blocks',len(report['blocks']),'tables',len(report['tables']),'sections',len(report['sections']))
