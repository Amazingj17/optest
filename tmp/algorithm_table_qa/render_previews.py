from pathlib import Path
import importlib.util
import tempfile
import os
import time
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageChops

BASE = Path(r'D:\programing\python\optest\tmp\algorithm_table_qa')
tempfile.tempdir = str(BASE)
renderer_path = r'C:\Users\17441\.codex\plugins\cache\openai-primary-runtime\documents\26.905.11957\skills\documents\render_docx.py'
spec = importlib.util.spec_from_file_location('docx_renderer', renderer_path)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
def replace_with_retry(source, destination):
    for attempt in range(10):
        try:
            return os.replace(source, destination)
        except PermissionError:
            if attempt == 9: raise
            time.sleep(0.2)
renderer.replace = replace_with_retry

def png_pages(pdf_path, dpi, output_folder, **kwargs):
    doc = pdfium.PdfDocument(pdf_path)
    paths = []
    for i in range(len(doc)):
        page = doc[i]
        bitmap = page.render(scale=dpi / 72)
        path = Path(output_folder) / f'page-raw-{i+1}.png'
        image = bitmap.to_pil()
        image.save(path)
        image.close()
        paths.append(str(path))
        bitmap.close(); page.close()
    doc.close()
    return paths

renderer.convert_from_path = png_pages
for variant in ('before', 'after'):
    renderer.convert_to_pdf = lambda *args, _v=variant, **kwargs: (str(BASE / f'{_v}.pdf'), 'Word native PDF export')
    source = r'D:\programing\python\optest\docs\项目说明书.docx' if variant == 'before' else r'D:\programing\python\optest\docs\项目说明书_算法三线表版.docx'
    paths = renderer.rasterize(source, str(BASE / variant), 144, False, False)
    print(variant, len(paths), 'pages rendered through packaged renderer with Word/PDFium adapters')
    for start in range(0, len(paths), 12):
        montage = Image.new('RGB', (1200, 1230), 'white')
        draw = ImageDraw.Draw(montage)
        for j, path in enumerate(paths[start:start+12]):
            with Image.open(path) as im:
                im.thumbnail((280, 370))
                x, y = (j%4)*300, (j//4)*410
                montage.paste(im, (x+10, y+25))
                draw.text((x+12, y+6), f'Page {start+j+1}', fill='black')
        montage.save(BASE / f'{variant}_overview_{start+1}.png')

changes = []
for page in range(1, 41):
    with Image.open(BASE / f'before/page-{page}.png') as a, Image.open(BASE / f'after/page-{page}.png') as b:
        if ImageChops.difference(a.convert('RGB'), b.convert('RGB')).getbbox(): changes.append(page)
print('Visually changed pages:', changes)
