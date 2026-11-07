"""00b_convert.py — raw files -> <data root>/interim parquet (never writes to the raw files).
Reads: raw/smit_figshare/*.RDATA, (if present) Dunford CSV, dog DogMoveData.csv
Writes: interim/smit_accel.parquet, smit_anno.parquet, dunford_cats.parquet, dog_move.parquet
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyreadr
from config import paths as P

P.ensure_dirs()


def smit():
    for name, out in (("anno_data.RDATA", "smit_anno.parquet"), ("accel_data.RDATA", "smit_accel.parquet")):
        dst = P.INTERIM / out
        if dst.exists():
            print("atla", out); continue
        res = pyreadr.read_r(str(P.SMIT_RAW / name))
        print(name, "nesneler:", {k: v.shape for k, v in res.items()})
        for k, df in res.items():
            for c in df.select_dtypes("category").columns:
                df[c] = df[c].astype(str)
            df.to_parquet(dst, index=False)
            print("yazıldı", dst, df.shape)


def dunford():
    src = next(P.DUNFORD_RAW.glob("*.csv"), None)
    dst = P.INTERIM / "dunford_cats.parquet"
    if src is None or dst.exists():
        print("Dunford: atla/yok"); return
    pd.read_csv(src).to_parquet(dst, index=False)
    print("yazıldı", dst)


def dog():
    src = next(P.DOG_RAW.rglob("DogMoveData.csv"), None)
    dst = P.INTERIM / "dog_move.parquet"
    if src is None or dst.exists():
        print("Köpek: atla/yok"); return
    writer = None
    for ch in pd.read_csv(src, chunksize=1_000_000, dtype={"DogID": "int16", "TestNum": "int16"}):
        for c in ("ABack_x", "ABack_y", "ABack_z", "ANeck_x", "ANeck_y", "ANeck_z", "GBack_x", "GBack_y", "GBack_z", "GNeck_x", "GNeck_y", "GNeck_z"):
            if c in ch: ch[c] = ch[c].astype("float32")
        t = pa.Table.from_pandas(ch, preserve_index=False)
        writer = writer or pq.ParquetWriter(dst, t.schema)
        writer.write_table(t)
    writer.close(); print("yazıldı", dst)


if __name__ == "__main__":
    smit(); dunford(); dog()
