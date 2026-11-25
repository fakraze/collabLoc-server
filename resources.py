"""Download and verify CollabLoc resources using only the Python standard library.

Usage: python3 resources.py {download,verify}
Paths are relative to this script. Downloads do not install Python packages.
"""
import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESOURCES = ({'id': 'netvlad',
  'url': 'https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat',
  'path': 'third_party/netvlad/VGG16-NetVLAD-Pitts30K.mat',
  'sha256': 'a67d9d897d3b7942f206478e3a22a4c4c9653172ae2447041d35f6cb278fdc67'},
 {'id': 'raft_archive',
  'url': 'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip',
  'path': '.cache/resources/raft-official.zip',
  'sha256': '4be6101b271f58ec49866da5cf609fd17e86e9cae2483f70630ef4a295dc66bd',
  'extracted_path': 'RAFT/models/raft-small.pth',
  'extracted_sha256': 'c7d41b9cc88442bb8aa911dbb33086dac55a226394b142937ff22d5578717332',
  'member': 'models/raft-small.pth'},
 {'id': 'pycolmap',
  'url': 'https://files.pythonhosted.org/packages/12/8c/5a07ea4a036015d86702bcb224bb8c2f4d35868ed3c2a9ca1fdac76c361f/pycolmap-0.0.1-cp37-cp37m-manylinux2014_x86_64.whl',
  'path': '.cache/resources/pycolmap-0.0.1-cp37-cp37m-manylinux2014_x86_64.whl',
  'sha256': '1b336835b64e869143f6206fcdad591b6bfaabfdac7470bc8ee9839d5cad7e58'},
 {'id': 'superpoint',
  'url': 'https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/weights/superpoint_v1.pth',
  'path': 'third_party/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth',
  'sha256': '52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e'})


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches(path, expected):
    return path.is_file() and sha256(path) == expected


def verify():
    errors = []
    for resource in RESOURCES:
        checks = [(resource["path"], resource["sha256"])]
        if "extracted_path" in resource:
            checks.append((resource["extracted_path"], resource["extracted_sha256"]))
        for name, expected in checks:
            path = ROOT / name
            if not path.is_file():
                errors.append("Missing: " + name)
            elif not matches(path, expected):
                errors.append("Checksum mismatch: " + name)
            else:
                print("Verified", name)
    if errors:
        raise SystemExit("\n".join(errors) + "\nRun: python3 resources.py download")
    print("PASS: all downloadable resources match SHA-256.")


def download_file(url, path, expected):
    if matches(path, expected):
        print("Verified", path.relative_to(ROOT))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        print("Downloading", url, flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "CollabLoc-resources/2"})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        if not matches(temporary, expected):
            raise ValueError("Checksum mismatch: " + str(path))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def download():
    for resource in RESOURCES:
        path = ROOT / resource["path"]
        download_file(resource["url"], path, resource["sha256"])
        if "extracted_path" not in resource:
            continue
        target = ROOT / resource["extracted_path"]
        if matches(target, resource["extracted_sha256"]):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        try:
            with zipfile.ZipFile(path) as archive:
                with archive.open(resource["member"]) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
            if not matches(temporary, resource["extracted_sha256"]):
                raise ValueError("Extracted weight checksum mismatch: " + str(target))
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
    verify()
    print("The pycolmap wheel was downloaded only; install dependencies using collabloc.yml.")
    print("raft-small-old.pth has no public URL; retain your existing copy when using the original commands.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "download"))
    args = parser.parse_args()
    try:
        {"verify": verify, "download": download}[args.command]()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, "Resource error: {}\n".format(error))


if __name__ == "__main__":
    main()
