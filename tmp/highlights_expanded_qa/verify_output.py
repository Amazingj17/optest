from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
from lxml import etree as E
import pypdfium2 as pdfium
from PIL import Image,ImageDraw

BASE=Path(r'D:\programing\python\optest')
QA=BASE/'tmp/highlights_expanded_qa'
ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math','w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
with ZipFile(BASE/'docs/项目说明书_修订版.docx') as a,ZipFile(BASE/'docs/项目说明书_亮点详解版.docx') as b:
    r1=E.fromstring(a.read('word/document.xml'));r2=E.fromstring(b.read('word/document.xml'))
    m1=[''.join(m.xpath('.//m:t/text()',namespaces=ns))for m in r1.xpath('.//m:oMath',namespaces=ns)]
    m2=[''.join(m.xpath('.//m:t/text()',namespaces=ns))for m in r2.xpath('.//m:oMath',namespaces=ns)]
    assert m1==m2 and len(m2)==7
    old_media={sha256(a.read(n)).hexdigest()for n in a.namelist()if n.startswith('word/media/')}
    new_media={sha256(b.read(n)).hexdigest()for n in b.namelist()if n.startswith('word/media/')}
    assert old_media.issubset(new_media)
    assert len(r2.xpath('.//wp:inline',namespaces=ns))-len(r1.xpath('.//wp:inline',namespaces=ns))==4
    toc=''.join(r2.xpath('.//w:sdt//w:t/text()',namespaces=ns))
    for i in range(1,6):assert f'1.3.{i}' in toc
    body1=r1.find('w:body',ns);body2=r2.find('w:body',ns)
    def remaining_text(body):
        inside=False;words=[]
        for e in body:
            if e.tag.endswith('sdt'):continue
            s=''.join(e.xpath('.//w:t/text()',namespaces=ns))
            if s=='1.3 项目亮点':inside=True
            if s=='第 2 章 技术理论':inside=False
            if not inside:words.append(s)
        return ''.join(words)
    assert remaining_text(body1)==remaining_text(body2),'Text outside section 1.3 changed'
print('Verified five subsection contents entries, four new inline images, unchanged math/media and text outside 1.3.')

out=QA/'after_word';out.mkdir(exist_ok=True)
pdf=pdfium.PdfDocument(QA/'final_word.pdf')
paths=[];number_edges=[]
for i in range(len(pdf)):
    page=pdf[i];tp=page.get_textpage();s=tp.get_text_range()
    if '1.3.' in s or '图1-' in s.replace(' ',''):
        print('Highlight/figure page',i+1,[(label,label in s)for label in ['1.3.1','1.3.2','1.3.3','1.3.4','1.3.5']])
    for n in range(1,8):
        search=tp.search(f'（2-{n}）');match=search.get_next();search.close()
        if match:
            begin,count=match;boxes=[tp.get_charbox(j)for j in range(begin,begin+count)]
            if min(b[0]for b in boxes)>400:
                number_edges.append(max(b[2]for b in boxes))
                print('Equation',n,'page',i+1)
    bitmap=page.render(scale=2);im=bitmap.to_pil()
    path=out/f'page-{i+1}.png';im.save(path);paths.append(path)
    im.close();bitmap.close();tp.close();page.close()
print('Word pages',len(pdf))
pdf.close()
assert len(number_edges)==7
assert max(number_edges)-min(number_edges)<0.1
print('All seven formula number right edges aligned.')
for start in range(0,len(paths),8):
    sheet=Image.new('RGB',(1600,1060),'white');draw=ImageDraw.Draw(sheet)
    for j,path in enumerate(paths[start:start+8]):
        with Image.open(path) as im:
            im.thumbnail((380,490));x=(j%4)*400;y=(j//4)*530
            sheet.paste(im,(x+10,y+25));draw.text((x+10,y+6),f'Page {start+j+1}',fill='black')
    sheet.save(out/f'overview-{start+1}.png')
