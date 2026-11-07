"""All paths live here; no other module hard-codes a path.

The data root is read from the environment variable ``KEDI_DATA_ROOT`` (default: ``./data`` next to the repository).
Expected layout under the data root: ``raw/`` (downloads, never modified), ``interim/`` (derived files), ``env/``, ``tmp/``.
Only code and small result tables are written inside the repository.
"""
import os
from pathlib import Path

SEED = 20261105

CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("KEDI_DATA_ROOT", CODE_ROOT / "data")).expanduser()

THIRD_PARTY = CODE_ROOT / "third_party"
SMIT_CODE = THIRD_PARTY / "Cat_Accelerometry_zip"  # Smit's R code (zip fallback; git clone fails on Windows because of ":" in a file name)
REPORTS = CODE_ROOT / "results" / "reports"
TABLES = CODE_ROOT / "results" / "tables"
FIGURES = CODE_ROOT / "results" / "figures"
LOGS = CODE_ROOT / "logs"

RAW = DATA_ROOT / "raw"
SMIT_RAW = RAW / "smit_figshare"
DUNFORD_RAW = RAW / "dunford_dryad"
DOG_RAW = RAW / "vehkaoja_mendeley"
INTERIM = DATA_ROOT / "interim"
ENV = DATA_ROOT / "env"

MANIFEST = TABLES / "data_manifest.csv"

# Dunford / dog 1-s features and intermediate files
FEATURES = INTERIM / "features"
DUNFORD_FEAT = FEATURES / "dunford_1s.parquet"
DUNFORD_FEAT40 = FEATURES / "dunford_1s_native40.parquet"   # sampling-rate sensitivity only
DOG_FEAT = FEATURES / "dog_aneck_1s.parquet"
DUNFORD_CSV = INTERIM / "dunford_cats.parquet"
DOG_PARQUET = INTERIM / "dog_move.parquet"
R05 = INTERIM / "r05"

# Cache / temporary files stay under the data root
TMP = DATA_ROOT / "tmp"


def ensure_dirs():
    for p in (REPORTS, TABLES, FIGURES, LOGS, SMIT_RAW, DUNFORD_RAW, DOG_RAW, INTERIM, TMP):
        p.mkdir(parents=True, exist_ok=True)


R06 = INTERIM / "r06"
DUNFORD_FEAT_V2 = FEATURES / "dunford_v2_1s.parquet"
R08 = INTERIM / "r08"   # R08: DM / DR intermediates (dog K-L1/K-L2 posteriors, bag parquet files)
