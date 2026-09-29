"""One-command offline demo launcher; paths resolve relative to this file."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import webbrowser


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Run search + HRL and open the result viewer')
    parser.add_argument('--no-open', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scene', action='append')
    args = parser.parse_args()
    try:
        import torch, numpy, yaml, pandas, networkx
    except ImportError as error:
        print(f'Missing dependency: {error}\nInstall first:\n'
              f'  "{sys.executable}" -m pip install -e "{root / "src"}[train,analysis,dev]"')
        return 1
    output = (args.output or root / 'demo/runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    command = [sys.executable, str(root / 'src/scripts/run_hybrid_demo.py'),
               '--bundle', str(root / 'demo'), '--output', str(output)]
    for scene in args.scene or []:
        command += ['--scene', scene]
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    code = subprocess.call(command, cwd=root / 'src', env=env)
    if code == 0 and not args.no_open:
        webbrowser.open((output / 'index.html').as_uri())
    return code


if __name__ == '__main__':
    raise SystemExit(main())
