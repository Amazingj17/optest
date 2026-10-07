from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E

BASE = Path(r'D:\programing\python\optest')
SOURCE = BASE / 'docs/项目说明书.docx'
PREVIOUS = BASE / 'docs/项目说明书_算法三线表版.docx'
OUTPUT = BASE / 'docs/项目说明书_修订版.docx'
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
MC='http://schemas.openxmlformats.org/markup-compatibility/2006'
WP='http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
NS={'w':W,'m':M,'mc':MC,'wp':WP}
def q(s): return f'{{{W}}}{s}'
def el(s, **attrs):
    e=E.Element(q(s))
    for key,value in attrs.items():e.set(q(key),str(value))
    return e
def text(p):return ''.join(p.xpath('.//w:t/text()',namespaces=NS))
def ppr(p):
    e=p.find('w:pPr',NS)
    if e is None:e=el('pPr');p.insert(0,e)
    return e
def setpr(p, name, **attrs):
    pp=ppr(p)
    for old in pp.findall('w:'+name,NS):pp.remove(old)
    pp.append(el(name,**attrs))

with ZipFile(SOURCE) as zin:
    root=E.fromstring(zin.read('word/document.xml'))
    body=root.find('w:body',NS)
    math_before=[E.tostring(m,method='c14n') for m in root.xpath('.//m:oMath',namespaces=NS)]
    original_math_paras=[e for e in body if e.find('m:oMathPara',NS) is not None]
    assert len(original_math_paras)==7
    anchors={}
    for p in list(body):
        for ac in p.findall('.//mc:AlternateContent',NS):
            a=ac.find('.//wp:anchor',NS)
            if a is None:continue
            tx=a.find('.//w:txbxContent',NS)
            if tx is None:continue
            number=text(tx)
            assert number in [f'（2-{n}）' for n in range(1,8)],number
            anchors[number]=deepcopy(tx.find('w:p',NS))
            parent=ac.getparent()
            parent.remove(ac)
            if parent.tag==q('r') and len(parent)==0:parent.getparent().remove(parent)
    assert len(anchors)==7

    for index,math_p in enumerate(original_math_paras,1):
        table=el('tbl')
        tp=el('tblPr');tp.append(el('tblW',w=9404,type='dxa'))
        tp.append(el('jc',val='center'));tp.append(el('tblLayout',type='fixed'))
        tb=el('tblBorders')
        for side in ('top','left','bottom','right','insideH','insideV'):tb.append(el(side,val='nil'))
        tp.append(tb)
        margins=el('tblCellMar')
        for side in ('top','left','bottom','right'):margins.append(el(side,w=120 if side in ('top','bottom') else 0,type='dxa'))
        tp.append(margins);table.append(tp)
        widths=[900,7604,900]
        grid=el('tblGrid')
        for width in widths:grid.append(el('gridCol',w=width))
        table.append(grid)
        tr=el('tr');rp=el('trPr');rp.append(el('cantSplit'));tr.append(rp);table.append(tr)
        for col,width in enumerate(widths):
            tc=el('tc');tcp=el('tcPr');tcp.append(el('tcW',w=width,type='dxa'))
            tcp.append(el('vAlign',val='center'));tc.append(tcp);tr.append(tc)
            if col==1:
                p=deepcopy(math_p)
                setpr(p,'keepNext',val=0)
                setpr(p,'spacing',before=0,after=0,line=240,lineRule='auto')
            elif col==2:
                p=deepcopy(anchors[f'（2-{index}）'])
                setpr(p,'jc',val='right')
                setpr(p,'spacing',before=0,after=0,line=240,lineRule='auto')
                setpr(p,'ind',firstLine=0,firstLineChars=0,left=0,right=0)
                setpr(p,'snapToGrid',val=0)
                for r in p.findall('w:r',NS):
                    rpr=r.find('w:rPr',NS)
                    if rpr is None:rpr=el('rPr');r.insert(0,rpr)
                    for name in ('rFonts','sz','szCs'):
                        for old in rpr.findall('w:'+name,NS):rpr.remove(old)
                    rpr.append(el('rFonts',ascii='Times New Roman',hAnsi='Times New Roman',eastAsia='宋体'))
                    rpr.append(el('sz',val=21));rpr.append(el('szCs',val=21))
            else:
                p=el('p');setpr(p,'spacing',before=0,after=0,line=240,lineRule='auto')
            tc.append(p)
        where=body.index(math_p);body.remove(math_p);body.insert(where,table)

    # Carry forward the already approved algorithm three-line table.
    with ZipFile(PREVIOUS) as prev:
        prevroot=E.fromstring(prev.read('word/document.xml'))
        algorithm=next(t for t in prevroot.findall('.//w:tbl',NS) if '清空旧候选' in text(t))
        old=next(p for p in body if p.tag==q('p') and '清空旧候选' in text(p))
        caption=old.getprevious();where=body.index(caption)
        body.remove(caption);body.remove(old);body.insert(where,deepcopy(algorithm))

    heading_source=next(p for p in body if text(p)=='1.2 项目背景')
    paragraph_source=next(p for p in body if text(p).startswith('项目据此确定了三项工程需求'))
    chapter2=next(p for p in body if text(p)=='第 2 章 技术理论')
    insert_at=body.index(chapter2)
    def paragraph(label, content, heading=False):
        p=el('p')
        source=heading_source if heading else paragraph_source
        p.append(deepcopy(ppr(source)))
        for value,bold in ((label,not heading),(content,False)):
            if not value:continue
            r=el('r')
            if bold:rp=el('rPr');rp.append(el('b'));r.append(rp)
            t=el('t');t.text=value;r.append(t);p.append(r)
        return p
    additions=[
        paragraph('','1.3 项目亮点',True),
        paragraph('','项目将离线 DAG 调度中的任务依赖、异构资源和通信开销纳入统一环境，形成从分层策略训练、候选改进到调度验证与结果追溯的完整流程，主要亮点如下。'),
        paragraph('（1）分层决策与合法性约束协同。','高层 LSTM 编码当前任务特征序列并选择就绪任务，低层 GAT 编码任务条件下的资源图并分配执行节点。ready_mask 与 node_mask 分别约束两层动作，使任务选择和资源分配遵循同一套依赖、设备和容量规则。'),
        paragraph('（2）启发式先验与有界学习残差结合。','将 HEFT upward rank 与预计完成时间 EFT 分别作为两层先验，通过 tanh 限制神经网络对决策偏好的修正幅度，并配合分阶段冻结和验证选模，使训练过程与初始启发式能力能够分别追踪。'),
        paragraph('（3）独立候选改进与统一择优。','搜索使用实际关键链、关键块修复和前缀快照复用，改善候选调度并减少重复模拟。HEFT、独立搜索和冻结 HRL 分别生成完整方案，经同一模拟器校验后按 makespan 择优，再逐动作重放选中方案，保留选择来源与决策记录。'),
        paragraph('（4）可复现评测与跨系统运行。','按基础 DAG 隔离数据划分，在固定场景和三个随机种子下记录配置、权重及逐场景指标；统一策略接口支持方法对照，演示页面支持时间线查看和结果导出，并已形成 openEuler、openKylin 虚拟机的 CPU 短训练与验证记录。'),
        paragraph('','正式批次在固定 108 个验证场景、54 个基础 DAG 上取得平均归一化 makespan 0.918223，相对 HEFT 降低约 8.18%，归档调度合法率为 100%。该结果对应候选组合方法，学习模块的独立表现及组合计算开销见第 4 章。'),
    ]
    for offset,p in enumerate(additions):body.insert(insert_at+offset,p)
    math_after=[E.tostring(m,method='c14n') for m in root.xpath('.//m:oMath',namespaces=NS)]
    assert math_after==math_before,'Native equations changed'
    assert not root.xpath('.//w:txbxContent',namespaces=NS)
    for n in range(1,8):
        assert sum(text(t)==f'（2-{n}）' for t in root.xpath('.//w:tbl/w:tr/w:tc',namespaces=NS))==1
    xml=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
    with ZipFile(OUTPUT,'w') as zout:
        for info in zin.infolist():zout.writestr(info,xml if info.filename=='word/document.xml' else zin.read(info.filename))
print('Added section 1.3; aligned all 7 editable equations and numbers; retained algorithm three-line table.')
print(OUTPUT)
