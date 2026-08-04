"""Measure the response times required by NFR 6.3.2 / test 10.2.3.

Runs a full lesson flow end to end, several times, and times every key action
the requirement names: loading the lesson content, analysing the learner's
speech and producing feedback, and closing the lesson.

Outputs
    docs/response-times.csv   raw measurements, one row per call
    docs/response-times.png   figure for the project book
    a summary table on the console

Usage
    # backend must be running:  uvicorn app.main:app --reload
    python scripts/measure_response_times.py \
        --email you@example.com --password **** \
        --audio ../LearningApp/app/src/main/res/raw/sample.wav --runs 5

Each run issues one /asr call per sentence in the lesson, and every /asr call
reaches Azure, so keep --runs modest.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO.parent / "docs"
GOOGLE_SERVICES = REPO.parent / "LearningApp" / "app" / "google-services.json"
SIGN_IN_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"

THRESHOLD_SEC = 3.0      # NFR 6.3.2
TARGET_RATIO = 0.80      # at least 80% of actions must stay under the threshold

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# Action labels: English for the figure, Hebrew for the book table.
ACTIONS = {
    "start_lesson":  ("Start lesson", "התחלת שיעור"),
    "load_content":  ("Load lesson content", "טעינת תוכן השיעור"),
    "assess_speech": ("Analyse speech + produce feedback", "ניתוח דיבור והפקת משוב"),
    "complete":      ("Complete lesson + final score", "סיום שיעור וחישוב ציון סופי"),
}

samples: dict[str, list[float]] = {k: [] for k in ACTIONS}
rows: list[dict] = []


def sign_in(email: str, password: str) -> str:
    key = json.loads(GOOGLE_SERVICES.read_text(encoding="utf-8"))["client"][0]["api_key"][0]["current_key"]
    r = httpx.post(SIGN_IN_URL, params={"key": key},
                   json={"email": email, "password": password, "returnSecureToken": True},
                   timeout=20)
    r.raise_for_status()
    return r.json()["idToken"]


def timed(action: str, run: int, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    samples[action].append(dt)
    rows.append({"run": run, "action": action,
                 "label_en": ACTIONS[action][0], "seconds": round(dt, 4)})
    flag = "" if dt <= THRESHOLD_SEC else "  <-- over threshold"
    print(f"    {ACTIONS[action][0]:<34} {dt:6.2f}s{flag}")
    return result


def pick_lesson(base: str, hdr: dict, match: str | None = None) -> str:
    """First lesson that has sentences, or the one containing `match`.

    Passing --match matters for a representative measurement: Azure runs in
    scripted mode, so audio that does not correspond to the reference sentence
    comes back as NoMatch and the request fails early, never reaching the
    feedback model — which is exactly the slow part we want to measure.
    """
    cats = httpx.get(base + "/categories/", headers=hdr, timeout=30).json()
    fallback = None
    for cat in cats:
        detail = httpx.get(f"{base}/categories/{cat['id']}", headers=hdr, timeout=30).json()
        for lesson in detail.get("lessons", []):
            if match is None:
                return lesson["id"]
            sentences = httpx.get(f"{base}/lessons/{lesson['id']}/sentences",
                                  headers=hdr, timeout=30).json()
            if fallback is None and sentences:
                fallback = lesson["id"]
            for s in sentences:
                if match.lower() in s["text"].lower():
                    print(f"  matched sentence: {s['text']!r}")
                    return lesson["id"]
    if match is not None:
        print(f"  no sentence matched {match!r} — falling back to the first lesson")
    if fallback:
        return fallback
    raise SystemExit("No lesson with content found.")


def run_once(base: str, hdr: dict, lesson_id: str, audio: Path, run: int) -> None:
    print(f"\n  Run {run}")
    run_id = timed("start_lesson", run, lambda: httpx.post(
        f"{base}/lessons/{lesson_id}/start", headers=hdr, timeout=60).json()["run_id"])

    sentences = timed("load_content", run, lambda: httpx.get(
        f"{base}/lessons/{lesson_id}/sentences", headers=hdr, timeout=60).json())

    blob = audio.read_bytes()
    for sentence in sentences:
        def call(sid=sentence["id"]):
            return httpx.post(
                base + "/asr", headers=hdr, timeout=120,
                files={"file": ("attempt.wav", blob, "audio/wav")},
                data={"sentence_id": sid, "run_id": run_id},
            )
        r = timed("assess_speech", run, call)
        if r.status_code != 200:
            print(f"      note: /asr returned HTTP {r.status_code} — timing still recorded")

    timed("complete", run, lambda: httpx.post(
        f"{base}/lessons/{lesson_id}/complete", headers=hdr,
        json={"run_id": run_id}, timeout=60))


def pct_under(values: list[float]) -> float:
    return 100.0 * sum(1 for v in values if v <= THRESHOLD_SEC) / len(values)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[idx]


def report() -> tuple[int, int]:
    print("\n" + "=" * 78)
    print(f"  {'Action':<34}{'n':>4}{'median':>9}{'p80':>9}{'max':>9}{'<=3s':>9}")
    print("-" * 78)
    for key, (label_en, _) in ACTIONS.items():
        v = samples[key]
        if not v:
            continue
        print(f"  {label_en:<34}{len(v):>4}{statistics.median(v):>8.2f}s"
              f"{percentile(v, 0.8):>8.2f}s{max(v):>8.2f}s{pct_under(v):>8.1f}%")
    every = [x for v in samples.values() for x in v]
    ok = sum(1 for x in every if x <= THRESHOLD_SEC)
    print("-" * 78)
    print(f"  {'ALL KEY ACTIONS':<34}{len(every):>4}{statistics.median(every):>8.2f}s"
          f"{percentile(every, 0.8):>8.2f}s{max(every):>8.2f}s{pct_under(every):>8.1f}%")
    print("=" * 78)
    verdict = "MET" if pct_under(every) >= TARGET_RATIO * 100 else "NOT MET"
    print(f"\n  Requirement 6.3.2 — at least {TARGET_RATIO:.0%} of key actions "
          f"within {THRESHOLD_SEC:.0f}s:  {verdict}")
    return ok, len(every)


def _open_unlocked(path: Path, **kwargs):
    """Open `path`, falling back to a timestamped name when it is locked.

    Measurement runs are expensive (every /asr call reaches Azure), so a file
    left open in a spreadsheet must never cost a whole run.
    """
    try:
        return path, path.open("w", **kwargs)
    except PermissionError:
        alt = path.with_name(f"{path.stem}-{time.strftime('%H%M%S')}{path.suffix}")
        print(f"  {path.name} is locked — writing {alt.name} instead")
        return alt, alt.open("w", **kwargs)


def write_outputs() -> None:
    DOCS.mkdir(exist_ok=True)
    csv_path, fh = _open_unlocked(DOCS / "response-times.csv",
                                  newline="", encoding="utf-8")
    with fh:
        w = csv.DictWriter(fh, fieldnames=["run", "action", "label_en", "seconds"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  raw measurements -> {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping the figure")
        return

    keys = [k for k in ACTIONS if samples[k]]
    labels = [ACTIONS[k][0].replace(" + ", "\n+ ") for k in keys]
    data = [samples[k] for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))

    bp = ax1.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for patch in bp["boxes"]:
        patch.set_facecolor("#DCEEFB")
        patch.set_edgecolor("#2E5FA3")
    for median in bp["medians"]:
        median.set_color("#17324F")
        median.set_linewidth(2)
    for i, vals in enumerate(data, start=1):
        ax1.scatter([i] * len(vals), vals, s=16, color="#2E5FA3",
                    alpha=0.55, zorder=3, linewidths=0)
    ax1.axhline(THRESHOLD_SEC, color="#C0392B", linestyle="--", linewidth=1.6,
                label=f"requirement: {THRESHOLD_SEC:.0f} s")
    ax1.set_ylabel("Response time (seconds)")
    ax1.set_title("Response time per key action", fontweight="bold")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(axis="y", alpha=0.25)
    ax1.tick_params(axis="x", labelsize=8.5)

    ratios = [pct_under(samples[k]) for k in keys]
    colors = ["#2E8B57" if r >= TARGET_RATIO * 100 else "#C0392B" for r in ratios]
    bars = ax2.barh(labels, ratios, color=colors, height=0.55)
    ax2.axvline(TARGET_RATIO * 100, color="#17324F", linestyle="--", linewidth=1.6,
                label=f"target: {TARGET_RATIO:.0%}")
    for bar, r in zip(bars, ratios):
        ax2.text(min(r + 2, 96), bar.get_y() + bar.get_height() / 2,
                 f"{r:.0f}%", va="center", fontsize=9, fontweight="bold")
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("Share of measurements within the threshold (%)")
    ax2.set_title("Compliance with the 3-second requirement", fontweight="bold")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.grid(axis="x", alpha=0.25)
    ax2.tick_params(axis="y", labelsize=8.5)

    fig.suptitle("SpeakUp — System Response Times (requirement 6.3.2)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = DOCS / "response-times.png"
    try:
        fig.savefig(png, dpi=200, facecolor="white")
    except PermissionError:
        png = png.with_name(f"response-times-{time.strftime('%H%M%S')}.png")
        fig.savefig(png, dpi=200, facecolor="white")
    print(f"  figure           -> {png}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8000")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--audio", required=True, help="a recorded attempt (wav/m4a)")
    p.add_argument("--match", default=None,
                   help="test the lesson containing this sentence text")
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        print(f"Audio file not found: {audio}")
        return 2

    base = args.base.rstrip("/")
    hdr = {"Authorization": f"Bearer {sign_in(args.email, args.password)}"}
    lesson_id = pick_lesson(base, hdr, args.match)
    print(f"Backend: {base}\nLesson under test: {lesson_id}\nRuns: {args.runs}")

    for i in range(1, args.runs + 1):
        run_once(base, hdr, lesson_id, audio, i)

    ok, total = report()
    write_outputs()
    return 0 if ok / total >= TARGET_RATIO else 1


if __name__ == "__main__":
    sys.exit(main())
