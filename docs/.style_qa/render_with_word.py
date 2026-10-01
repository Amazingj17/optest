import importlib.util,sys,subprocess,shutil
from pathlib import Path
SKILL=Path(r'C:\Users\17441\.codex\plugins\cache\openai-primary-runtime\documents\26.905.11957\skills\documents')
spec=importlib.util.spec_from_file_location('renderer',SKILL/'render_docx.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
def convert(input_path,user_profile,convert_tmp_dir,stem,verbose=False):
    pdf=Path(convert_tmp_dir)/(Path(input_path).stem+'.pdf')
    result=subprocess.run(['powershell.exe','-NoProfile','-File',str(Path(__file__).with_name('export_word.ps1')),'-InputDoc',str(Path(input_path).resolve()),'-OutputPdf',str(pdf)],capture_output=True,text=True)
    print(result.stdout);print(result.stderr)
    if result.returncode:raise RuntimeError('Word PDF export failed')
    return str(pdf),result.stdout
renderer.convert_to_pdf=convert
poppler=Path(r'C:\Users\17441\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin')
orig_info=renderer.pdfinfo_from_path
orig_convert=renderer.convert_from_path
renderer.pdfinfo_from_path=lambda *a,**kw:orig_info(*a,poppler_path=str(poppler),**kw)
renderer.convert_from_path=lambda *a,**kw:orig_convert(*a,poppler_path=str(poppler),**kw)
renderer.main()
