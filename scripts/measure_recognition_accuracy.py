"""Measure pronunciation-recognition accuracy for NFR 6.3.1 / test 10.2.1.

Ground truth comes from the file names in the recordings folder:

    "Could I see the menu.m4a"          -> the sentence was pronounced correctly
    "WRONG Could I see the menu.m4a"    -> a deliberate mistake was recorded

Every recording is sent through the real /asr endpoint, against the sentence
whose text matches the file name, and the analysis the system produced is
compared with that label.

Metrics (requirement 6.3.1)
    accuracy          correctly classified recordings / all recordings   >= 85%
    detection rate    flagged mistakes / recordings that contain one     >= 80%
    false positives   flagged clean recordings / clean recordings        <= 10%

Optional: --labels labels.json lets you name the word that was mispronounced
in each WRONG recording, e.g. {"WRONG Is this dish spicy": "spicy"}. When a
label is given, the run also reports whether the system blamed the right word.

Usage
    python scripts/measure_recognition_accuracy.py \
        --email you@example.com --password **** \
        --dir "C:/Users/User/Documents/Sound Recordings"
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO.parent / "docs"
GOOGLE_SERVICES = REPO.parent / "LearningApp" / "app" / "google-services.json"
SIGN_IN_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"

ACC_TARGET, DET_TARGET, FP_LIMIT = 85.0, 80.0, 10.0
WRONG_PREFIX = "wrong "

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def norm(text: str) -> str:
    """Compare sentences ignoring punctuation, case and spacing."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def sign_in(email: str, password: str) -> str:
    key = json.loads(GOOGLE_SERVICES.read_text(encoding="utf-8"))["client"][0]["api_key"][0]["current_key"]
    r = httpx.post(SIGN_IN_URL, params={"key": key},
                   json={"email": email, "password": password, "returnSecureToken": True},
                   timeout=30)
    r.raise_for_status()
    return r.json()["idToken"]


def load_sentence_index(base: str, hdr: dict) -> dict[str, tuple[str, str]]:
    """normalized sentence text -> (sentence_id, original text)"""
    index: dict[str, tuple[str, str]] = {}
    for cat in httpx.get(base + "/categories/", headers=hdr, timeout=60).json():
        detail = httpx.get(f"{base}/categories/{cat['id']}", headers=hdr, timeout=60).json()
        for lesson in detail.get("lessons", []):
            for s in httpx.get(f"{base}/lessons/{lesson['id']}/sentences",
                               headers=hdr, timeout=60).json():
                index[norm(s["text"])] = (s["id"], s["text"])
    return index


def flagged_errors(body: dict) -> list[str]:
    """Short human-readable list of everything the system flagged."""
    out = []
    out += [f"missing:{w}" for w in body.get("missing_words", [])]
    out += [f"extra:{w}" for w in body.get("extra_words", [])]
    out += [f"sub:{s.get('expected')}->{s.get('heard')}" for s in body.get("substitutions", [])]
    out += [f"mispron:{w.get('word')}" for w in body.get("mispronounced_words", [])]
    out += [f"silent:{e.get('word')}" for e in body.get("silent_letter_errors", [])]
    out += [f"hardsoftc:{e.get('word')}" for e in body.get("hard_soft_c_errors", [])]
    out += [f"cluster:{e.get('word')}" for e in body.get("cluster_errors", [])]
    return out


def blamed_words(body: dict) -> set[str]:
    words = set(w.lower() for w in body.get("missing_words", []))
    words |= {(s.get("expected") or "").lower() for s in body.get("substitutions", [])}
    for key in ("mispronounced_words", "silent_letter_errors",
                "hard_soft_c_errors", "cluster_errors"):
        words |= {(e.get("word") or "").lower() for e in body.get(key, [])}
    return {w for w in words if w}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8000")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--dir", required=True, help="folder holding the recordings")
    p.add_argument("--labels", default=None, help="optional JSON: stem -> expected word")
    args = p.parse_args()

    folder = Path(args.dir)
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in (".m4a", ".wav", ".mp3"))
    if not files:
        print(f"No recordings found in {folder}")
        return 2

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8")) if args.labels else {}

    base = args.base.rstrip("/")
    hdr = {"Authorization": f"Bearer {sign_in(args.email, args.password)}"}
    print(f"Backend: {base}\nRecordings: {len(files)}\nIndexing sentences...")
    index = load_sentence_index(base, hdr)
    print(f"  {len(index)} sentences available\n")

    rows, skipped = [], []
    for f in files:
        stem = f.stem
        has_mistake = stem.lower().startswith(WRONG_PREFIX)
        spoken = stem[len(WRONG_PREFIX):] if has_mistake else stem
        key = norm(spoken)
        if key not in index:
            skipped.append((f.name, "no matching sentence in the database"))
            continue
        sentence_id, sentence_text = index[key]

        t0 = time.perf_counter()
        r = httpx.post(base + "/asr", headers=hdr, timeout=180,
                       files={"file": (f.name, f.read_bytes(), "audio/mp4")},
                       data={"sentence_id": sentence_id,
                             "run_id": "00000000-0000-0000-0000-000000000000"})
        dt = time.perf_counter() - t0

        if r.status_code != 200:
            skipped.append((f.name, f"HTTP {r.status_code}: {r.text[:90]}"))
            print(f"  [SKIP] {f.name}  -> HTTP {r.status_code}")
            continue

        body = r.json()
        flags = flagged_errors(body)
        system_says_mistake = bool(flags)
        correct = system_says_mistake == has_mistake

        expected_word = labels.get(stem)
        right_word = None
        if expected_word and system_says_mistake:
            right_word = expected_word.lower() in blamed_words(body)

        rows.append({
            "file": f.name,
            "sentence": sentence_text,
            "ground_truth": "mistake" if has_mistake else "clean",
            "score": body.get("final_score"),
            "passed": body.get("is_passed"),
            "system_flagged": "yes" if system_says_mistake else "no",
            "classification": "correct" if correct else "incorrect",
            "flags": "; ".join(flags) if flags else "-",
            "expected_word": expected_word or "",
            "blamed_expected_word": "" if right_word is None else ("yes" if right_word else "no"),
            "seconds": round(dt, 2),
        })
        mark = "OK " if correct else "MISS"
        print(f"  [{mark}] {f.name:<44} score={body.get('final_score'):>3}  "
              f"flags={len(flags)}  ({dt:.1f}s)")

    if not rows:
        print("\nNothing was analysed.")
        for name, why in skipped:
            print(f"  skipped {name}: {why}")
        return 2

    clean = [r for r in rows if r["ground_truth"] == "clean"]
    faulty = [r for r in rows if r["ground_truth"] == "mistake"]
    tp = sum(1 for r in faulty if r["system_flagged"] == "yes")
    fp = sum(1 for r in clean if r["system_flagged"] == "yes")
    accuracy = 100.0 * sum(1 for r in rows if r["classification"] == "correct") / len(rows)
    detection = 100.0 * tp / len(faulty) if faulty else float("nan")
    fp_rate = 100.0 * fp / len(clean) if clean else float("nan")

    print("\n" + "=" * 74)
    print(f"  recordings analysed          {len(rows)}  ({len(clean)} clean, {len(faulty)} with a mistake)")
    print(f"  classification accuracy      {accuracy:5.1f}%   (requirement: >= {ACC_TARGET:.0f}%)"
          f"   {'MET' if accuracy >= ACC_TARGET else 'NOT MET'}")
    print(f"  mistake detection rate       {detection:5.1f}%   (requirement: >= {DET_TARGET:.0f}%)"
          f"   {'MET' if detection >= DET_TARGET else 'NOT MET'}")
    print(f"  false positive rate          {fp_rate:5.1f}%   (requirement: <= {FP_LIMIT:.0f}%)"
          f"   {'MET' if fp_rate <= FP_LIMIT else 'NOT MET'}")
    print("=" * 74)
    if len(clean) < 10:
        print(f"  NOTE: with {len(clean)} clean recordings the smallest measurable false-positive")
        print(f"        rate above zero is {100/len(clean):.1f}%, so the {FP_LIMIT:.0f}% limit can only")
        print( "        be met with zero false positives. Report this alongside the number.")

    if skipped:
        print("\n  not analysed:")
        for name, why in skipped:
            print(f"    {name}: {why}")

    DOCS.mkdir(exist_ok=True)
    out = DOCS / "recognition-accuracy.csv"
    try:
        fh = out.open("w", newline="", encoding="utf-8-sig")
    except PermissionError:
        out = out.with_name(f"recognition-accuracy-{time.strftime('%H%M%S')}.csv")
        fh = out.open("w", newline="", encoding="utf-8-sig")
    with fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  per-recording detail -> {out}")

    ok = accuracy >= ACC_TARGET and detection >= DET_TARGET and fp_rate <= FP_LIMIT
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
