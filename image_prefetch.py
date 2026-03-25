from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def _target_path(cache_dir: Path, url: str) -> Path:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix or ".img"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}{suffix}"


def _download_one(url: str, target_path: Path, timeout_sec: int) -> Tuple[str, Path | None]:
    if target_path.exists() and target_path.stat().st_size > 0:
        return url, target_path

    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=timeout_sec) as resp, tmp_path.open("wb") as out:
            out.write(resp.read())
        os.replace(tmp_path, target_path)
        return url, target_path
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        return url, None


def prefetch_item_images(
    meta_map: Dict[str, Dict],
    resolve_image_fn: Callable[[Dict], str],
    cache_dir: Path,
    max_workers: int = 16,
    timeout_sec: int = 8,
) -> Dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    unique_urls = []
    seen = set()
    for meta in meta_map.values():
        image_url = (resolve_image_fn(meta) or "").strip()
        if image_url.startswith(("http://", "https://")) and image_url not in seen:
            seen.add(image_url)
            unique_urls.append(image_url)

    if not unique_urls:
        return {}

    print(f"[ImagePrefetch] start downloading {len(unique_urls)} urls -> {cache_dir}")
    url_to_path: Dict[str, str] = {}
    done = 0
    ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_download_one, url, _target_path(cache_dir, url), timeout_sec): url
            for url in unique_urls
        }
        for future in as_completed(future_map):
            done += 1
            url, local_path = future.result()
            if local_path is not None:
                ok += 1
                url_to_path[url] = str(local_path)
            if done % 1000 == 0 or done == len(unique_urls):
                print(f"[ImagePrefetch] progress {done}/{len(unique_urls)} (ok={ok})")

    return url_to_path
