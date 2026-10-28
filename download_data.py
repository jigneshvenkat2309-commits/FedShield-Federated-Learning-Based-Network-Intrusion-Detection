"""
Download and validate the NSL-KDD benchmark dataset.
Files:
- KDDTrain+.txt (Training set)
- KDDTest+.txt  (Held-out test set)
"""

import os
import hashlib
import urllib.request
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

URLS = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
}

EXPECTED_LINE_COUNTS = {
    "KDDTrain+.txt": 125973,
    "KDDTest+.txt": 22544
}

def compute_hashes(filepath):
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
    return md5_hash.hexdigest(), sha256_hash.hexdigest()

def download_and_validate():
    os.makedirs(DATA_DIR, exist_ok=True)
    summary = {}
    
    for filename, url in URLS.items():
        dest = os.path.join(DATA_DIR, filename)
        print(f"Downloading {filename} from {url}...")
        
        req = urllib.request.Request(url, headers={"User-Agent": "FedShield-Pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as out_file:
            if resp.status != 200:
                raise RuntimeError(f"HTTP error {resp.status} while downloading {url}")
            data = resp.read()
            out_file.write(data)
            
        size_bytes = os.path.getsize(dest)
        md5_hex, sha256_hex = compute_hashes(dest)
        
        # Verify line count
        with open(dest, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f if line.strip()]
            line_count = len(lines)
            
        # Verify columns on first and last line
        first_cols = len(lines[0].split(","))
        last_cols = len(lines[-1].split(","))
        
        expected_lines = EXPECTED_LINE_COUNTS.get(filename)
        match_expected = (line_count == expected_lines)
        
        summary[filename] = {
            "url": url,
            "filepath": dest,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "line_count": line_count,
            "expected_line_count": expected_lines,
            "line_count_valid": match_expected,
            "columns_first_row": first_cols,
            "columns_last_row": last_cols,
            "md5": md5_hex,
            "sha256": sha256_hex
        }
        
        print(f"  [OK] {filename}: {size_bytes} bytes ({summary[filename]['size_mb']} MB), {line_count} rows, {first_cols} cols")
        print(f"       MD5:    {md5_hex}")
        print(f"       SHA256: {sha256_hex}")
        if not match_expected:
            print(f"  [WARNING] Line count {line_count} does not match expected {expected_lines}!")

    return summary

if __name__ == "__main__":
    download_and_validate()
