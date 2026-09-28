"""Verify the sole supported local Zenodo release; never download another source."""
from __future__ import annotations
import argparse
from pathlib import Path


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument('--raw-root',default='data/raw'); args=parser.parse_args()
    target=Path(args.raw_root)/'zenodo-18927122-derived'
    if not target.is_dir():
        raise FileNotFoundError(f'Place the Zenodo 18927122 release at {target}; automatic downloads are disabled.')
    if not (target/'system_configs.tar.xz').is_file():
        raise FileNotFoundError(f'{target} is missing system_configs.tar.xz')
    if not tuple(target.glob('rnc*_json.tar.xz')):
        raise FileNotFoundError(f'{target} has no rnc workflow archives')
    print(f'Using local Zenodo release only: {target.resolve()}')
if __name__=='__main__': main()
