import argparse
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('outdir')
args = parser.parse_args()
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(outdir)
skill = Path(r'C:\Users\17441\.codex\plugins\cache\openai-primary-runtime\documents\26.905.11957\skills\documents')
spec = importlib.util.spec_from_file_location('renderer', skill / 'render_docx.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
# The user explicitly confirmed installed LibreOffice for this revision.
renderer._resolve_soffice = lambda: r'D:\AAA\LibreOffice\program\soffice.exe'

def raster_pdf(pdf_path, dpi, output_folder, **kwargs):
    pdf = pdfium.PdfDocument(pdf_path)
    files = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=dpi/72)
        image = bitmap.to_pil()
        path = Path(output_folder) / f'page-render-{i+1}.png'
        image.save(path)
        image.close(); bitmap.close(); page.close()
        files.append(str(path))
    pdf.close()
    return files

def safe_replace(source, destination):
    for attempt in range(10):
        try: return os.replace(source, destination)
        except PermissionError:
            if attempt == 9: raise
            time.sleep(0.2)

renderer.convert_from_path = raster_pdf
renderer.replace = safe_replace
pages = renderer.rasterize(args.input, str(outdir), 144, True, True)
print('RENDERED', len(pages), 'pages')
for start in range(0, len(pages), 8):
    overview = Image.new('RGB', (1600, 1060), 'white')
    draw = ImageDraw.Draw(overview)
    for j, path in enumerate(pages[start:start+8]):
        with Image.open(path) as im:
            im.thumbnail((380, 490))
            x,y = (j%4)*400,(j//4)*530
            overview.paste(im,(x+10,y+25))
            draw.text((x+10,y+6),f'Page {start+j+1}',fill='black')
    overview.save(outdir / f'overview-{start+1}.png')
