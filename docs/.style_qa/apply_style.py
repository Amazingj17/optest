from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from copy import deepcopy as cp
import re,json,hashlib

BASE=Path(__file__).parent
REF=Path(r'D:\programing\python\optest归档\决赛作品提交模板\操作系统开源创新大赛项目说明书.docx')
SRC=BASE.parent/'项目说明书_双系统验证更新版.docx'
OUT=BASE.parent/'项目说明书_双系统验证更新版_模板样式.docx'
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS={'w':W,'r':R,'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing','a':'http://schemas.openxmlformats.org/drawingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
q=lambda s:'{'+W+'}'+s
def new(tag,**attrs):return E.Element(q(tag),{q(k):str(v) for k,v in attrs.items()})
def text(n):return ''.join(n.xpath('.//w:t/text()',namespaces=NS))
def setprop(n,tag,**attrs):
    for old in n.findall(q(tag)):n.remove(old)
    c=new(tag,**attrs);n.append(c);return c
def serialize(n):return E.tostring(n,xml_declaration=True,encoding='UTF-8',standalone=True)
srcz=ZipFile(SRC);refz=ZipFile(REF)
parts={n:srcz.read(n) for n in srcz.namelist()}
root=E.fromstring(parts['word/document.xml']);body=root.find(q('body'));old=list(body)
refroot=E.fromstring(refz.read('word/document.xml'));refbody=refroot.find(q('body'))
styles=E.fromstring(parts['word/styles.xml']);refstyles=E.fromstring(refz.read('word/styles.xml'))
refmap={s.get(q('styleId')):'Ref'+s.get(q('styleId')) for s in refstyles.findall(q('style'))}
def remap(n):
    for x in n.iter():
        if E.QName(x).localname in ('pStyle','rStyle','tblStyle','basedOn','next','link') and x.get(q('val')) in refmap:x.set(q('val'),refmap[x.get(q('val'))])
    return n
for s in refstyles.findall(q('style')):
    c=remap(cp(s));c.set(q('styleId'),refmap[s.get(q('styleId'))]);c.find(q('name')).set(q('val'),'模板 '+s.find(q('name')).get(q('val')))
    # Numbering is rendered by preserved source text, never inherited implicitly.
    for x in c.xpath('.//w:numPr',namespaces=NS):x.getparent().remove(x)
    styles.append(c)
defaults=styles.find(q('docDefaults'));styles.remove(defaults)
styles.insert(0,cp(refstyles.find(q('docDefaults'))))
# Keep real built-in source style names for Word navigation / TOC.
for sid,rsid in [('a','1'),('1','2'),('2','3'),('3','4')]:
    dest=next(s for s in styles.findall(q('style')) if s.get(q('styleId'))==sid)
    source=next(s for s in refstyles.findall(q('style')) if s.get(q('styleId'))==rsid)
    for tag in ('pPr','rPr','basedOn'):
        for x in dest.findall(q(tag)):dest.remove(x)
        x=source.find(q(tag))
        if x is not None:dest.append(cp(x))
    for x in dest.xpath('.//w:numPr',namespaces=NS):x.getparent().remove(x)
    if sid!='a':setprop(dest,'basedOn',val='a')
    ppr=dest.find(q('pPr'))
    if ppr is not None and sid!='a':
        setprop(ppr,'ind',firstLine=0,firstLineChars=0,left=0)
        setprop(ppr,'keepNext');setprop(ppr,'keepLines')
    rpr=dest.find(q('rPr'))
    if rpr is not None:
        setprop(rpr,'color',val='000000')
        for x in list(rpr):
            if E.QName(x).namespace!=W:rpr.remove(x)
        if sid in ('1','2'):setprop(rpr,'b',val=0);setprop(rpr,'bCs',val=0)
# Heading 2/3 no inherited template styles after remapping: materialize needed fonts/spacing.
for sid,ea,size in [('2','黑体',28),('3','宋体',24)]:
    s=next(s for s in styles.findall(q('style')) if s.get(q('styleId'))==sid)
    setprop(s.find(q('rPr')),'rFonts',ascii='Times New Roman',hAnsi='Times New Roman',eastAsia=ea)
    p=s.find(q('pPr'));setprop(p,'spacing',before=120,after=120,line=400,lineRule='exact');setprop(p,'jc',val='left')
# Imported blank headers and matching page-number footer.
rels=E.fromstring(parts['word/_rels/document.xml.rels']);ct=E.fromstring(parts['[Content_Types].xml'])
RELNS='http://schemas.openxmlformats.org/package/2006/relationships'
CTNS='http://schemas.openxmlformats.org/package/2006/content-types'
ridmap={}
for i,name in enumerate(['word/header1.xml','word/header2.xml','word/header3.xml','word/footer1.xml','word/footer2.xml','word/footer3.xml','word/footer4.xml','word/footer5.xml','word/footer6.xml']):
    target='word/template_'+Path(name).name;parts[target]=serialize(remap(E.fromstring(refz.read(name))))
    rid='rIdTemplate'+str(i+1);ridmap[name]=rid
    typ='header' if 'header' in name else 'footer'
    rel=E.SubElement(rels,'{'+RELNS+'}Relationship',Id=rid,Type=R+'/'+typ,Target=Path(target).name)
    E.SubElement(ct,'{'+CTNS+'}Override',PartName='/'+target,ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.'+typ+'+xml')
def section(kind):
    s=new('sectPr')
    E.SubElement(s,q('headerReference'),{q('type'):'default','{'+R+'}id':ridmap['word/header1.xml']})
    foot='word/footer3.xml' if kind in ('cover','toc') else 'word/footer5.xml'
    E.SubElement(s,q('footerReference'),{q('type'):'default','{'+R+'}id':ridmap[foot]})
    s.append(new('pgSz',w=12240,h=15840));s.append(new('pgMar',top=1701,right=1418,bottom=1134,left=1418,header=720,footer=720,gutter=0))
    if kind=='body':s.append(new('pgNumType',start=1))
    s.append(new('cols',space=720,num=1));s.append(new('docGrid',linePitch=360,charSpace=0))
    return s
def paragraph(value='',style=None):
    p=new('p');pr=new('pPr');p.append(pr)
    if style:pr.append(new('pStyle',val=style))
    if value:r=new('r');t=new('t');t.text=value;r.append(t);p.append(r)
    return p
def replace_text(p,value):
    runs=p.findall(q('r'));rp=cp(runs[0].find(q('rPr'))) if runs and runs[0].find(q('rPr')) is not None else None
    for x in list(p):
        if x.tag!=q('pPr'):p.remove(x)
    r=new('r');p.append(r)
    if rp is not None:r.append(rp)
    t=new('t');t.text=value;r.append(t)
for n in list(body):body.remove(n)
# Exact retained-template cover components.
for i in [0,1,2]:body.append(remap(cp(refbody[i])))
cover=remap(cp(refbody[3]));body.append(cover)
for tr,value in zip(cover.findall(q('tr')),['CPN-HRL-DAG','','2026年10月1日']):
    p=tr.findall(q('tc'))[1].find(q('p'));replace_text(p,value)
    pr=p.find(q('pPr'));setprop(pr,'jc',val='center');setprop(pr,'ind',firstLine=0,firstLineChars=0)
endcover=paragraph();endcover.find(q('pPr')).append(section('cover'));body.append(endcover)
toc_title=remap(cp(refbody[6]));setprop(toc_title.find(q('pPr')),'outlineLvl',val=9);body.append(toc_title)
toc=E.SubElement(body,q('sdt'));spr=E.SubElement(toc,q('sdtPr'));E.SubElement(spr,q('tag'),{q('val'):'TemplateTOC'})
content=E.SubElement(toc,q('sdtContent'));tp=paragraph();content.append(tp)
for typ in ['begin','separate','end']:
    r=new('r');tp.append(r)
    if typ=='begin':
        r.append(new('fldChar',fldCharType='begin'));r=new('r');tp.append(r);ins=new('instrText');ins.set('{http://www.w3.org/XML/1998/namespace}space','preserve');ins.text=' TOC \\o "1-3" \\h \\z \\u ';r.append(ins)
    else:r.append(new('fldChar',fldCharType=typ))
endtoc=paragraph();endtoc.find(q('pPr')).append(section('toc'));body.append(endtoc)
metadata=[cp(old[17]),cp(old[19])]
source_nodes=[cp(n) for n in old[21:331]]
for n in source_nodes:
    if n.tag==q('p'):
        value=text(n).strip();pr=n.find(q('pPr'))
        if pr is None:pr=new('pPr');n.insert(0,pr)
        sid=pr.find(q('pStyle'));sid=sid.get(q('val')) if sid is not None else None
        level=None
        if re.match(r'^[1-6]\.\d+\.\d+\s',value):level=3
        elif re.match(r'^[1-6]\.\d+\s',value):level=2
        elif re.match(r'^[1-6]\s+\S',value) or value=='绪论':level=1
        elif sid in ('1','2','3'):level=int(sid)
        if level:
            for x in list(pr):pr.remove(x)
            pr.append(new('pStyle',val=str(level)))
            if level==1:
                chapter=1 if value=='绪论' else int(value[0]);title=value if chapter==1 else re.sub(r'^\d+\s+','',value)
                replace_text(n,f'第 {chapter} 章 {title}')
                if chapter!=1:pr.append(new('pageBreakBefore'))
            # Direct font/color overrides must not defeat the template headings.
            for r in n.findall(q('r')):
                rp=r.find(q('rPr'))
                if rp is not None:r.remove(rp)
        else:
            # Preserve technical prose and breaks; only normalize layout properties.
            for tag in ['pStyle','pBdr','shd','spacing','ind','jc','rPr','numPr','pageBreakBefore','sectPr','keepNext']:
                for x in pr.findall(q(tag)):pr.remove(x)
            pr.insert(0,new('pStyle',val='a'));pr.append(new('jc',val='both'))
            is_picture=bool(n.xpath('.//w:drawing',namespaces=NS));is_math=bool(n.xpath('.//m:oMath',namespaces=NS))
            is_caption=bool(re.match(r'^(表\d+-\d+|算法\d+-\d+)\s',value))
            is_code=('\n' in ''.join(n.itertext()) and n.xpath('.//w:br',namespaces=NS)) or value.startswith(('resources:','输入：','python ','uv ','pip '))
            if is_picture or is_math:
                setprop(pr,'jc',val='center');pr.append(new('ind',firstLine=0,firstLineChars=0));setprop(pr,'spacing',before=120,after=120,line=240,lineRule='auto')
                pr.append(new('keepNext'))
            elif is_caption:
                setprop(pr,'jc',val='center');pr.append(new('ind',firstLine=0,firstLineChars=0));pr.append(new('keepNext'));setprop(pr,'spacing',before=120,after=100,line=300,lineRule='auto')
            elif is_code:
                setprop(pr,'jc',val='left');pr.append(new('ind',firstLine=0,firstLineChars=0));setprop(pr,'spacing',before=80,after=80,line=300,lineRule='auto')
            # Only Word text runs: native OMML formatting stays intact.
            for r in n.findall(q('r')):
                rp=r.find(q('rPr'))
                if rp is None:rp=new('rPr');r.insert(0,rp)
                for tag in ['rFonts','color','sz','szCs','rStyle','smallCaps','shd']:
                    for x in rp.findall(q(tag)):rp.remove(x)
                rp.append(new('rFonts',ascii='Consolas' if is_code else 'Times New Roman',hAnsi='Consolas' if is_code else 'Times New Roman',eastAsia='宋体'))
                rp.append(new('color',val='000000'));rp.append(new('sz',val=21 if (is_code or is_caption) else 24));rp.append(new('szCs',val=21 if (is_code or is_caption) else 24))
    body.append(n)
    if n.tag==q('p') and text(n).startswith('正式实验采用固定的 108'):
        for m in metadata:body.append(m)
body.append(section('body'))
# Fit and normalize all source tables, without touching the cloned cover table.
for table in body.findall(q('tbl')):
    if table is cover:continue
    pr=table.find(q('tblPr'));grid=table.find(q('tblGrid'))
    widths=[int(c.get(q('w'))) for c in grid];factor=9404/sum(widths)
    scaled=[round(w*factor) for w in widths];scaled[-1]+=9404-sum(scaled)
    for c,wid in zip(grid,scaled):c.set(q('w'),str(wid))
    for tag in ['tblStyle','tblBorders','tblInd','tblW','shd','tblCellSpacing']:
        for x in pr.findall(q(tag)):pr.remove(x)
    pr.append(new('tblW',w=9404,type='dxa'));pr.append(new('jc',val='center'));pr.append(new('tblInd',w=0,type='dxa'))
    borders=new('tblBorders');pr.append(borders)
    for edge in ['top','left','bottom','right','insideH','insideV']:borders.append(new(edge,val='single',sz=4,color='BFBFBF'))
    for rowidx,row in enumerate(table.findall(q('tr'))):
        rowpr=row.find(q('trPr'))
        if rowpr is None:rowpr=new('trPr');row.insert(0,rowpr)
        for x in rowpr.findall(q('trHeight')):rowpr.remove(x)
        for x in rowpr.findall(q('tblCellSpacing')):rowpr.remove(x)
        if rowidx==0:setprop(rowpr,'tblHeader')
        for cell in row.findall(q('tc')):
            cpr=cell.find(q('tcPr'));cw=cpr.find(q('tcW'))
            if cw is not None:cw.set(q('w'),str(round(int(cw.get(q('w')))*factor)))
            for tag in ['tcBorders','tcMar','shd']:
                for x in cpr.findall(q(tag)):cpr.remove(x)
            cpr.append(new('shd',val='clear',fill='F2F2F2' if rowidx==0 else 'FFFFFF'))
            setprop(cpr,'vAlign',val='center');mar=new('tcMar');cpr.append(mar)
            for edge,wid in [('top',70),('bottom',70),('left',85),('right',85)]:mar.append(new(edge,w=wid,type='dxa'))
            for p in cell.findall(q('p')):
                ppr=p.find(q('pPr'))
                if ppr is None:ppr=new('pPr');p.insert(0,ppr)
                for x in list(ppr):ppr.remove(x)
                ppr.append(new('pStyle',val='a'));ppr.append(new('ind',firstLine=0,firstLineChars=0));ppr.append(new('spacing',before=0,after=0,line=300,lineRule='atLeast'))
                if rowidx==0:ppr.append(new('keepNext'))
                for r in p.findall(q('r')):
                    rp=r.find(q('rPr'))
                    if rp is not None:r.remove(rp)
                    rp=new('rPr');r.insert(0,rp);rp.append(new('rFonts',ascii='Times New Roman',hAnsi='Times New Roman',eastAsia='宋体'));rp.append(new('sz',val=21));rp.append(new('szCs',val=21));rp.append(new('color',val='000000'))
                    if rowidx==0:rp.append(new('b'))
# Scale inline images to the new available width, preserving aspect ratio and media.
maxcx=9404*635
for drawing in body.xpath('.//wp:inline',namespaces=NS):
    ext=drawing.find('wp:extent',NS);cx=int(ext.get('cx'));cy=int(ext.get('cy'))
    if cx>maxcx:
        f=maxcx/cx;ext.set('cx',str(maxcx));ext.set('cy',str(round(cy*f)))
        for ex in drawing.xpath('.//a:xfrm/a:ext',namespaces=NS):ex.set('cx',str(maxcx));ex.set('cy',str(round(cy*f)))
settings=E.fromstring(parts['word/settings.xml']);setprop(settings,'updateFields',val='true')
# Avoid source screen-document headers and automatic font embedding on field refresh.
parts['word/document.xml']=serialize(root);parts['word/styles.xml']=serialize(styles);parts['word/settings.xml']=serialize(settings)
parts['word/_rels/document.xml.rels']=serialize(rels);parts['[Content_Types].xml']=serialize(ct)
with ZipFile(OUT,'w',ZIP_DEFLATED) as z:
    for name,data in parts.items():z.writestr(name,data)
# Integrity gates.
orig_tables=[text(t) for t in E.fromstring(srcz.read('word/document.xml')).xpath('//w:tbl',namespaces=NS)]
final_tables=[text(t) for t in root.xpath('//w:tbl',namespaces=NS) if t is not cover]
assert sorted(orig_tables)==sorted(final_tables)
assert len(root.xpath('//m:oMath',namespaces=NS))==len(E.fromstring(srcz.read('word/document.xml')).xpath('//m:oMath',namespaces=NS))
changed=[n for n in srcz.namelist() if parts[n]!=srcz.read(n)]
assert set(changed)=={'word/document.xml','word/styles.xml','word/settings.xml','word/_rels/document.xml.rels','[Content_Types].xml'}
assert hashlib.sha256(REF.read_bytes()).hexdigest()==json.loads((BASE/'reference_evidence.json').read_text(encoding='utf8'))['sha256']
print('Created',OUT,'; tables, formulas, images and preserve-only package parts verified; changed',changed)
