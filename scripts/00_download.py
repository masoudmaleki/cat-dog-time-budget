"""00_download.py — the three datasets + Smit's code. Idempotent; at most 2 attempts per source.
Reads: network. Writes: <data root>/raw/*, third_party/Cat_Accelerometry, results/tables/data_manifest.csv
"""
import sys, json, hashlib, shutil, subprocess, zipfile, datetime as dt
from pathlib import Path
from urllib.parse import quote
import requests
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import paths as P

P.ensure_dirs()
UA = {"User-Agent": "Mozilla/5.0 (research-download)"}
MANIFEST_ROWS, FAILS = [], []


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def stream(url, dest, expected_size=None, expected_md5=None, tries=2):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        ok_size = expected_size is None or dest.stat().st_size == expected_size
        ok_md5 = expected_md5 is None or md5(dest) == expected_md5
        if ok_size and ok_md5:
            print(f"  atla (var): {dest.name}")
            return True
    last = None
    for k in range(tries):
        try:
            with requests.get(url, stream=True, headers=UA, timeout=60, allow_redirects=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) or expected_size
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
                    for ch in r.iter_content(1 << 20):
                        f.write(ch); bar.update(len(ch))
            if expected_md5 and md5(tmp) != expected_md5:
                raise IOError("md5 uyuşmuyor")
            tmp.replace(dest)
            return True
        except Exception as e:
            last = e
            print(f"  deneme {k+1} başarısız: {e}")
    FAILS.append((url, str(last)))
    return False


def add_manifest(path, url):
    path = Path(path)
    MANIFEST_ROWS.append(dict(dosya=str(path), boyut=path.stat().st_size, sha256=sha256(path),
                              kaynak_url=url, indirilme_zamani=dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")))


def smit():
    print("== Smit (FigShare)")
    try:
        meta = requests.get("https://api.figshare.com/v2/articles/23605842", headers=UA, timeout=60).json()
    except Exception as e:
        FAILS.append(("figshare api", str(e))); return
    (P.SMIT_RAW / "figshare_article_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("  version:", meta.get("version"), "| doi:", meta.get("doi"), "| modified:", meta.get("modified_date"))
    for fi in meta["files"]:
        dest = P.SMIT_RAW / fi["name"]
        if stream(fi["download_url"], dest, fi["size"], fi.get("computed_md5")):
            add_manifest(dest, fi["download_url"])


def smit_code():
    print("== Smit kodu (GitHub)")
    dest = P.SMIT_CODE
    if (dest / ".git").exists() or (dest.exists() and any(dest.iterdir())):
        print("  atla (var)"); return
    url = "https://github.com/MSmit1992/Cat_Accelerometry"
    for k in range(2):
        r = subprocess.run(["git", "clone", url, str(dest)], capture_output=True, text=True)
        if r.returncode == 0:
            return
        print("  git clone başarısız:", r.stderr.strip()[:200])
    FAILS.append((url, "git clone 2 deneme başarısız"))


def dunford():
    print("== Dunford (Dryad)")
    try:
        doi = quote("doi:10.5061/dryad.q2bvq83sx", safe="")
        base = "https://datadryad.org"
        ds = requests.get(f"{base}/api/v2/datasets/{doi}", headers=UA, timeout=60)
        ds.raise_for_status(); ds = ds.json()
        ver = base + ds["_links"]["stash:version"]["href"]
        files = requests.get(ver + "/files", headers=UA, timeout=60); files.raise_for_status()
        files = files.json()["_embedded"]["stash:files"]
    except Exception as e:
        FAILS.append(("dryad api", str(e))); print("  API hata:", e); return
    for fi in files:
        dest = P.DUNFORD_RAW / fi["path"]
        url = base + fi["_links"]["stash:download"]["href"]
        if stream(url, dest, fi.get("size")):
            add_manifest(dest, url)


def dog():
    print("== Köpek (Mendeley)")
    got = None
    for v in (4, 3, 1):
        url = f"https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/vxhx934tbn-{v}.zip"
        dest = P.DOG_RAW / f"vxhx934tbn-{v}.zip"
        if stream(url, dest, tries=2):
            got = (dest, url); print("  sürüm", v); break
    if not got:
        return
    zdest, url = got
    add_manifest(zdest, url)
    marker = P.DOG_RAW / "_extracted.ok"
    if marker.exists():
        return
    with zipfile.ZipFile(zdest) as z:
        z.extractall(P.DOG_RAW)
    for inner in list(P.DOG_RAW.rglob("*.zip")):
        if inner == zdest:
            continue
        with zipfile.ZipFile(inner) as z:
            z.extractall(inner.parent)
    for f in P.DOG_RAW.rglob("*"):
        if f.is_file() and f.suffix != ".part" and f != zdest and f.name != "_extracted.ok":
            add_manifest(f, url + " (açılmış)")
    marker.write_text("ok")


if __name__ == "__main__":
    smit(); smit_code(); dunford(); dog()
    df = pd.DataFrame(MANIFEST_ROWS).drop_duplicates("dosya")
    df.to_csv(P.MANIFEST, index=False, encoding="utf-8")
    print(df[["dosya", "boyut"]].to_string())
    print("BAŞARISIZ:", FAILS)
