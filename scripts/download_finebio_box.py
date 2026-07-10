#!/usr/bin/env python3
"""Download FineBio dataset from Box shared link directly to scratch.

Uses headless Chromium (Playwright) on the server. Files stream to disk;
no need to load whole archives into memory. Safe to resume: skips existing files.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import requests

SHARED_LINK = "https://aist.box.com/s/9ai0p8rns31d2iok6z2xmv90st7gqtgs"
DEFAULT_PASSWORD = "54!bx%l(MZr]dCnOog"
DEFAULT_OUT = "/scratch/ll5914/Labos/Llava/data/FineBio"
DOWNLOAD_TIMEOUT_MS = 7_200_000  # 2 hours to trigger Box download
MAX_RETRIES = 5
CHUNK_SIZE = 8 * 1024 * 1024
ZIP_POLL_INTERVAL = 20
ZIP_POLL_MAX_WAIT = 3600  # 1 hour for Box to prepare folder zip


def parse_items(html: str) -> list[dict]:
    pattern = re.compile(
        r'"typedID":"([^"]+)","type":"([^"]+)","id":(\d+).*?"name":"([^"]+)".*?"itemSize":(\d+)'
    )
    return [
        {
            "typed_id": typed_id,
            "type": item_type,
            "id": item_id,
            "name": name,
            "size": int(size),
        }
        for typed_id, item_type, item_id, name, size in pattern.findall(html)
    ]


def human_size(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}PB"


def fetch_items() -> list[dict]:
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    resp = session.get(SHARED_LINK, timeout=60)
    resp.raise_for_status()
    items = parse_items(resp.text)
    if not items:
        raise RuntimeError("Could not parse file list from Box shared link")
    return items


def fetch_folder_items(folder_id: str) -> list[dict]:
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    resp = session.get(f"{SHARED_LINK.rstrip('/')}/folder/{folder_id}", timeout=60)
    resp.raise_for_status()
    return parse_items(resp.text)


def should_download(item: dict, only: set[str] | None) -> bool:
    if only is None:
        return True
    return item["name"] in only or item["id"] in only


def _session_from_context(context) -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    for cookie in context.cookies():
        session.cookies.set(
            cookie["name"],
            cookie["value"],
            domain=cookie.get("domain"),
            path=cookie.get("path", "/"),
        )
    return session


def _wait_for_zip_ready(session: requests.Session, url: str) -> None:
    if "zip_download" not in url:
        return
    elapsed = 0
    while elapsed < ZIP_POLL_MAX_WAIT:
        resp = session.head(url, timeout=60, allow_redirects=True)
        length = int(resp.headers.get("content-length", 0) or 0)
        if resp.status_code == 200 and length > 0:
            print(f"[ready] zip prepared ({human_size(length)})")
            return
        print(f"[wait] zip preparing... ({elapsed}s, status={resp.status_code})")
        time.sleep(ZIP_POLL_INTERVAL)
        elapsed += ZIP_POLL_INTERVAL
    raise TimeoutError(f"zip not ready after {ZIP_POLL_MAX_WAIT}s")


def _stream_to_file(session: requests.Session, url: str, dest: Path) -> None:
    _wait_for_zip_ready(session, url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    with session.get(url, stream=True, timeout=(60, 7200)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with tmp.open("wb") as handle:
            for chunk in resp.iter_content(CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (256 * 1024 * 1024) < CHUNK_SIZE:
                    pct = downloaded * 100 / total
                    print(
                        f"\r[dl] {dest.name}: {human_size(downloaded)}/{human_size(total)} ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
    tmp.rename(dest)
    if total:
        print()
    print(f"[ok] {dest.name} ({human_size(dest.stat().st_size)})")


def _get_cdn_url(page, url: str, label: str) -> str:
    page.goto(url, wait_until="networkidle", timeout=120_000)
    download_btn = page.get_by_role("button", name="Download", exact=True)
    if download_btn.count() == 0:
        raise RuntimeError(f"No Download button for {label}")

    with page.expect_download(timeout=DOWNLOAD_TIMEOUT_MS) as dl_info:
        download_btn.first.click()
    cdn_url = dl_info.value.url
    if not cdn_url:
        raise RuntimeError(f"No CDN URL returned for {label}")
    return cdn_url


def _download_with_retries(
    page, context, url: str, dest: Path, label: str, expected_size: int
) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} ({human_size(dest.stat().st_size)})")
        return

    print(f"[start] {label} ({human_size(expected_size)})")
    session = _session_from_context(context)
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            cdn_url = _get_cdn_url(page, url, label)
            print(f"[cdn] {cdn_url[:100]}...")
            _stream_to_file(session, cdn_url, dest)
            return
        except Exception as exc:
            last_err = exc
            part = dest.with_suffix(dest.suffix + ".part")
            if part.exists():
                part.unlink()
            if attempt < MAX_RETRIES:
                wait = 30 * attempt
                print(f"[retry] {label} attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                print(f"[retry] waiting {wait}s before retry...")
                time.sleep(wait)
    raise RuntimeError(f"Failed to download {label} after {MAX_RETRIES} attempts") from last_err


def download_file(page, context, item: dict, out_dir: Path) -> None:
    dest = out_dir / item["name"]
    url = f"{SHARED_LINK.rstrip('/')}/file/{item['id']}"
    _download_with_retries(page, context, url, dest, item["name"], item["size"])


def download_folder_zip(page, context, item: dict, out_dir: Path) -> None:
    dest = out_dir / f"{item['name']}.zip"
    url = f"{SHARED_LINK.rstrip('/')}/folder/{item['id']}"
    _download_with_retries(page, context, url, dest, f"folder {item['name']}", item["size"])


def download_folder_items(page, context, item: dict, out_dir: Path) -> None:
    """Download each file inside a folder individually (more reliable than zip)."""
    sub_dir = out_dir / item["name"]
    sub_dir.mkdir(parents=True, exist_ok=True)
    children = fetch_folder_items(item["id"])
    files = [c for c in children if c["type"] == "file"]
    subfolders = [c for c in children if c["type"] == "folder"]

    print(f"[folder] {item['name']}: {len(files)} file(s), {len(subfolders)} subfolder(s)")
    for child in sorted(files, key=lambda x: x["size"]):
        download_file(page, context, child, sub_dir)
    for child in subfolders:
        download_folder_items(page, context, child, sub_dir)


def download_with_playwright(
    password: str,
    items: list[dict],
    out_dir: Path,
    only: set[str] | None,
) -> list[str]:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [i for i in items if should_download(i, only)]
    failures: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.goto(SHARED_LINK, wait_until="networkidle", timeout=120_000)

        pwd_input = page.locator('input[type="password"]')
        if pwd_input.count() > 0:
            pwd_input.first.fill(password)
            page.keyboard.press("Enter")
            page.wait_for_load_state("networkidle", timeout=60_000)

        # Download zip files first (more reliable), then folders.
        files = sorted(
            [i for i in targets if i["type"] == "file"],
            key=lambda x: x["size"],
        )
        folders = sorted(
            [i for i in targets if i["type"] == "folder"],
            key=lambda x: x["size"],
        )

        for item in files:
            try:
                download_file(page, context, item, out_dir)
            except Exception as exc:
                msg = f"{item['name']}: {exc}"
                print(f"[fail] {msg}")
                failures.append(msg)

        for item in folders:
            try:
                download_folder_items(page, context, item, out_dir)
            except Exception as exc:
                msg = f"{item['name']}: {exc}"
                print(f"[fail] {msg}")
                failures.append(msg)

        browser.close()
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download FineBio Box dataset to scratch (server-side)"
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        help="Download only these filenames/IDs (e.g. annotations finebio_videos_fpv_train.zip)",
    )
    args = parser.parse_args()

    items = fetch_items()
    total = sum(item["size"] for item in items)
    print(f"Found {len(items)} items, total ~{human_size(total)}")
    for item in items:
        print(f"  [{item['type']:6}] {item['name']:<45} {human_size(item['size']):>8}")

    if args.list_only:
        return 0

    only = set(args.only) if args.only else None
    failures = download_with_playwright(args.password, items, Path(args.out_dir), only)
    if failures:
        print(f"\n[warn] {len(failures)} item(s) failed:")
        for msg in failures:
            print(f"  - {msg}")
        print("Re-run the script to retry failed items (completed files are skipped).")
        return 1

    print(f"\nDone. Data saved under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
