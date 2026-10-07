from pathlib import Path
import re
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

BASE=Path(r'D:\programing\python\optest\tmp\doc_revision_qa')
def inspect_pdf(path, outdir, render=False):
    outdir.mkdir(parents=True,exist_ok=True)
    pdf=pdfium.PdfDocument(path)
    print('PDF',path.name,'pages',len(pdf))
    boxes=[]
    for i in range(len(pdf)):
        p=pdf[i];tp=p.get_textpage();s=tp.get_text_range()
        if '1.3 项目亮点' in s: print('Highlight section/contents page',i+1)
        if render:
            bitmap=p.render(scale=2);im=bitmap.to_pil();im.save(outdir/f'page-{i+1}.png');im.close();bitmap.close()
        for number in range(1,8):
            label=f'（2-{number}）'
            search=tp.search(label)
            match=search.get_next()
            search.close()
            if match is None:continue
            start,count=match
            rects=[tp.get_charbox(n) for n in range(start,start+count)]
            left=min(b[0] for b in rects);bottom=min(b[1] for b in rects)
            right=max(b[2] for b in rects);top=max(b[3] for b in rects)
            if left<400:continue
            boxes.append((i+1,label,left,bottom,right,top))
            print('Equation number',i+1,label, 'rect',tuple(round(x,2) for x in (left,bottom,right,top)))
            im=Image.open(outdir/f'page-{i+1}.png')
            y=int((p.get_height()-(bottom+top)/2)*2)
            im.crop((100,max(0,y-65),im.width-100,min(im.height,y+65))).save(outdir/f'equation-{len(boxes)}.png')
            im.close()
        tp.close();p.close()
    pdf.close()
    return boxes

word_boxes=inspect_pdf(BASE/'final_word.pdf',BASE/'after_word',True)
lo_pdf=next((BASE/'after_lo').glob('*.pdf'))
lo_boxes=inspect_pdf(lo_pdf,BASE/'after_lo')
assert len(word_boxes)==7,(word_boxes,len(word_boxes))
assert len(lo_boxes)==7,(lo_boxes,len(lo_boxes))
for variant,boxes in [('Word',word_boxes),('LibreOffice',lo_boxes)]:
    right_edges=[b[4] for b in boxes]
    print(variant,'number right-edge spread (pt)',round(max(right_edges)-min(right_edges),3))
    collage=Image.new('RGB',(1100,1050),'white');draw=ImageDraw.Draw(collage)
    folder=BASE/('after_word' if variant=='Word' else 'after_lo')
    for j in range(7):
        with Image.open(folder/f'equation-{j+1}.png') as im:
            im.thumbnail((1080,130));collage.paste(im,(10,j*150+20))
        draw.text((10,j*150+2),f'Equation 2-{j+1}',fill='black')
    collage.save(BASE/f'equations-{variant}.png')
