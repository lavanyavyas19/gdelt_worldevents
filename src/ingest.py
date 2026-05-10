
import os
import io
import zipfile
import requests
from datetime import date, timedelta
from pathlib import Path

GDELT_BASE_URL = "http://data.gdeltproject.org/events/"


def _date_range(start: str, end: str):
    """Yield date objects from start to end (inclusive)."""
    current = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def download_day(day: date, out_dir: str, timeout: int = 120) -> str | None:
    """
    Download and extract one GDELT daily export.

    Returns path to extracted CSV, or None on failure.
    Skips if file already exists on disk.
    """
    date_str = day.strftime("%Y%m%d")
    csv_filename = f"{date_str}.export.CSV"
    zip_filename = f"{csv_filename}.zip"
    out_path = os.path.join(out_dir, csv_filename)

    if os.path.exists(out_path):
        return out_path

    url = GDELT_BASE_URL + zip_filename

    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  SKIP {date_str}: {e}")
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            members = zf.namelist()
            if not members:
                print(f"  SKIP {date_str}: empty zip")
                return None
            zf.extract(members[0], out_dir)
            extracted = os.path.join(out_dir, members[0])
            if extracted != out_path:
                os.rename(extracted, out_path)
    except (zipfile.BadZipFile, OSError) as e:
        print(f"  SKIP {date_str}: {e}")
        return None

    size_mb = os.path.getsize(out_path) / 1_000_000
    print(f"  {date_str}: {size_mb:.1f} MB")
    return out_path


def download_range(
    start: str,
    end: str,
    out_dir: str,
) -> list[str]:
    """
    Download all GDELT daily exports in [start, end].

    Parameters
    ----------
    start : ISO date string, e.g. "2025-12-01"
    end   : ISO date string, e.g. "2026-03-31"
    out_dir : directory to store extracted CSVs

    Returns
    -------
    List of paths to successfully downloaded/extracted CSVs.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    paths = []
    total_days = 0
    for day in _date_range(start, end):
        total_days += 1
        path = download_day(day, out_dir)
        if path:
            paths.append(path)

    print(f"\n  Downloaded {len(paths)}/{total_days} daily files → {out_dir}")
    return paths




def main():
    from .config import RAW_DIR

    print("=" * 60)
    print(" GDELT Data Ingestion")
    print(" Range: 2025-12-01 → 2026-03-31")
    print("=" * 60)

    download_range("2025-12-01", "2026-03-31", RAW_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
