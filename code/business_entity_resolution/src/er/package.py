"""
Build the final submission archive in the layout required by the challenge:

<team>_submission.zip
├── output/matching_results.tsv
├── output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md

Nothing is moved in the repository: files are read from where they live and
written into the zip under the required paths. Datasets are never included.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from .common import log

REPO = Path(__file__).resolve().parents[4]          # .../AmazonMLChallenge
PKG = REPO / "code" / "business_entity_resolution"


def run(out_dir: Path, team: str, zip_dir: Path | None = None) -> Path:
    zip_dir = zip_dir or REPO
    required = [out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv",
                PKG / "README.md", PKG / "requirements.txt", REPO / "Documentation_template.md"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing for the package: " + ", ".join(missing))
    path = zip_dir / f"{team}_submission.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        z.write(out_dir / "matching_results.tsv", "output/matching_results.tsv")
        z.write(out_dir / "candidate_pairs.tsv", "output/candidate_pairs.tsv")
        for f in sorted((PKG / "src").rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                z.write(f, "code/business_entity_resolution/" + f.relative_to(PKG).as_posix())
        for name in ("README.md", "requirements.txt", "kaggle_pipeline.ipynb", "__init__.py"):
            if (PKG / name).is_file():
                z.write(PKG / name, f"code/business_entity_resolution/{name}")
        z.write(REPO / "Documentation_template.md", "Documentation_template.md")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    log(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {len(names)} files)")
    for n in names:
        log(f"  {n}")
    return path
