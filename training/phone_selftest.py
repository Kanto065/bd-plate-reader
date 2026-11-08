"""Run photos through the installed app on a USB phone, without the photo picker.

Usage:
  python phone_selftest.py PHOTO_DIR [OUT_CSV]

It pushes every .jpg, .jpeg and .png in PHOTO_DIR to the app's own folder
(Android/data/com.bdplatereader/files/selftest), starts the app with the test
only link bdplate://selftest, waits for the SELFTEST_DONE line in logcat and
prints one row per photo. If PHOTO_DIR has a truth.csv (columns file,text, class
names in reading order) it also prints the exact match, the same way
pipeline_eval.py does, so phone and PC can be compared on the same photos.

Needs adb, a phone with USB debugging, and a build with the self test (build 23
or later). Nothing is sent anywhere: the photos go to the app's folder and the
results come back through logcat. The folder is emptied afterwards.
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PKG = "com.bdplatereader"
DEST = "/sdcard/Android/data/%s/files/selftest" % PKG
LINE = re.compile(r"'SELFTEST', '(\{.*\})'")


def find_adb():
    found = shutil.which("adb")
    if found:
        return found
    for root in (os.environ.get("ANDROID_HOME"), os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk")):
        if root:
            p = Path(root) / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb")
            if p.exists():
                return str(p)
    raise SystemExit("adb not found. Put platform-tools on PATH or set ANDROID_HOME.")


ADB = find_adb()


def adb(*args, check=False):
    r = subprocess.run([ADB, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise SystemExit("adb %s failed: %s" % (" ".join(args), (r.stderr or r.stdout).strip()))
    return r.stdout


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    folder = Path(sys.argv[1])
    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("phone_results.csv")
    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not photos:
        raise SystemExit("No photos in %s" % folder)
    if "\tdevice" not in adb("devices"):
        raise SystemExit("No phone found. Connect it, unlock it, and accept the USB debugging prompt.")

    adb("shell", "rm", "-rf", DEST)
    adb("shell", "mkdir", "-p", DEST, check=True)
    for p in photos:
        adb("push", str(p), DEST + "/" + p.name, check=True)
    adb("logcat", "-c")
    adb("shell", "am", "force-stop", PKG)
    adb("shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", "bdplate://selftest", PKG)

    rows, deadline = {}, time.time() + 60 + 6 * len(photos)
    while time.time() < deadline:
        log = adb("logcat", "-d", "-s", "ReactNativeJS:I")
        rows = {}
        for m in LINE.finditer(log):
            d = json.loads(m.group(1))
            rows[d["file"]] = d
        if "SELFTEST_DONE" in log:
            break
        time.sleep(3)
    else:
        print("Timed out, showing what arrived (%d of %d photos)." % (len(rows), len(photos)))
    adb("shell", "rm", "-rf", DEST)

    truth = {}
    if (folder / "truth.csv").exists():
        with open(folder / "truth.csv", encoding="utf-8", newline="") as f:
            truth = {r["file"]: r["text"].strip() for r in csv.DictReader(f)}
    scored = exact = 0
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "tokens", "text", "low", "reasons", "error", "ms", "truth", "match"])
        for p in photos:
            d = rows.get(p.name)
            if not d:
                w.writerow([p.name, "", "", "", "", "no result", "", truth.get(p.name, ""), ""])
                continue
            t = truth.get(p.name)
            match = ""
            if t is not None:
                scored += 1
                match = "yes" if d["tokens"] == t else "no"
                exact += match == "yes"
            w.writerow([p.name, d["tokens"], d["text"], d["low"], "; ".join(d["reasons"]), d["error"] or "", d["ms"], t or "", match])
            print("%-45s %-34s %s%s" % (p.name[:45], d["tokens"] or "(nothing)", "LOW " if d["low"] else "", match))
    print("photos %d, results %d" % (len(photos), len(rows)))
    if scored:
        print("scored %d: exact match %.1f%%" % (scored, 100 * exact / scored))
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
