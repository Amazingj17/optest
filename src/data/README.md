# Local dataset preparation

The raw dataset is not distributed in this Git repository. Obtain the release
from https://zenodo.org/records/18927122 and retain its attribution and licence.
The upstream dataset description is preserved in
[raw/zenodo-18927122-derived/README.md](raw/zenodo-18927122-derived/README.md).

For the default 50/100/300-task experiments, place the archives at:

```text
data/raw/zenodo-18927122-derived/
  system_configs.tar.xz
  rnc50_homo_json.tar.xz
  rnc50_hetero_json.tar.xz
  rnc100_homo_json.tar.xz
  rnc100_hetero_json.tar.xz
  rnc300_homo_json.tar.xz
  rnc300_hetero_json.tar.xz
```

The adapter reads archive members directly. Validate the local layout with:

```bash
python scripts/prepare_data.py
```

This command only checks local files and does not download data. Keep the
versioned `manifests/grapheonrl_mixed_iid_seed7.json` unchanged when reproducing
the reported split. Raw archives and extracted data are ignored by Git.
