"""Run the API-verifiable test scenarios from chapter 10 of the project book.

Covers the scenarios that can be observed from the server side:
    10.1.1  Select a Lesson from Categories
    10.1.3  Resume Incomplete Lesson
    10.1.4  View Achievements and Progress
    10.1.5  Sign In / Register        (token handling only)
    10.2.4  Authenticated and isolated access to learner data

Scenarios that need a physical device (audio playback, recording, lip sync,
profile screen) are listed at the end as NOT COVERED, so nothing is silently
reported as passing when it was never exercised.

Usage:
    # 1. start the backend:  uvicorn app.main:app --reload
    # 2. then:
    python scripts/run_book_test_scenarios.py --email you@example.com --password ****

The Firebase web API key is read from the Android app's google-services.json,
so no key has to be passed on the command line.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
GOOGLE_SERVICES = REPO.parent / "LearningApp" / "app" / "google-services.json"
SIGN_IN_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"

results: list[tuple[str, str, str, bool]] = []

# The scenario labels are in Hebrew; the Windows console defaults to cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def check(scenario: str, expected: str, actual: str, passed: bool) -> None:
    results.append((scenario, expected, actual, passed))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {scenario}\n         expected: {expected}\n         actual:   {actual}")


def firebase_api_key() -> str:
    data = json.loads(GOOGLE_SERVICES.read_text(encoding="utf-8"))
    return data["client"][0]["api_key"][0]["current_key"]


def sign_in(email: str, password: str) -> str:
    r = httpx.post(
        SIGN_IN_URL,
        params={"key": firebase_api_key()},
        json={"email": email, "password": password, "returnSecureToken": True},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["idToken"]


# ---------------------------------------------------------------- 10.1.5 + 10.2.4
def test_authentication(base: str, token: str) -> None:
    print("\n== 10.1.5 / 10.2.4 : authentication and access isolation ==")
    protected = ["/categories/", "/progress/overview", "/progress/badges"]

    for path in protected:
        r = httpx.get(base + path, timeout=20)
        check(
            f"בקשה ללא טוקן  ({path})",
            "הבקשה נדחית (401/403)",
            f"HTTP {r.status_code}",
            r.status_code in (401, 403),
        )

    for path in protected:
        r = httpx.get(base + path, headers={"Authorization": "Bearer not-a-real-token"}, timeout=20)
        check(
            f"בקשה עם טוקן פגום  ({path})",
            "הבקשה נדחית (401/403)",
            f"HTTP {r.status_code}",
            r.status_code in (401, 403),
        )

    r = httpx.get(base + "/categories/", headers=auth(token), timeout=20)
    check(
        "בקשה עם טוקן תקף",
        "הבקשה מתקבלת (200)",
        f"HTTP {r.status_code}",
        r.status_code == 200,
    )

    # The server must never take the user identity from the request itself.
    r = httpx.get(
        base + "/progress/overview",
        headers=auth(token),
        params={"user_id": "some-other-user-uid"},
        timeout=20,
    )
    baseline = httpx.get(base + "/progress/overview", headers=auth(token), timeout=20)
    same = r.status_code == 200 and r.json() == baseline.json()
    check(
        "הזרקת מזהה משתמש אחר בפרמטרים",
        "הנתונים המוחזרים זהים לנתוני המשתמש המאומת",
        "זהים" if same else "שונים / שגיאה",
        same,
    )


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------- 10.1.1
def test_category_and_lesson_selection(base: str, token: str) -> tuple[str, str]:
    print("\n== 10.1.1 : בחירת שיעור מתוך קטגוריות מוצעות ==")
    r = httpx.get(base + "/categories/", headers=auth(token), timeout=20)
    cats = r.json() if r.status_code == 200 else []
    check(
        "קיימות קטגוריות זמינות",
        "מוצגות כל הקטגוריות הזמינות לבחירה",
        f"{len(cats)} קטגוריות הוחזרו",
        r.status_code == 200 and len(cats) > 0,
    )

    complete = all(c.get("title") and c.get("description") for c in cats)
    check(
        "לכל קטגוריה מוצגים כותרת ותיאור",
        "לכל קטגוריה קיימים כותרת ותיאור",
        "כל הקטגוריות מלאות" if complete else "קיימת קטגוריה חסרת שדה",
        complete,
    )

    category_id = cats[0]["id"]
    r = httpx.get(f"{base}/categories/{category_id}", headers=auth(token), timeout=20)
    lessons = (r.json() or {}).get("lessons", []) if r.status_code == 200 else []
    check(
        "בחירת קטגוריה עם שיעורים",
        "מוצגת רשימת השיעורים השייכת לקטגוריה שנבחרה",
        f"{len(lessons)} שיעורים הוחזרו",
        r.status_code == 200 and len(lessons) > 0,
    )

    lesson_id = lessons[0]["id"]
    r = httpx.get(f"{base}/lessons/{lesson_id}", headers=auth(token), timeout=20)
    details = r.json() if r.status_code == 200 else {}
    r2 = httpx.get(f"{base}/lessons/{lesson_id}/sentences", headers=auth(token), timeout=20)
    sentences = r2.json() if r2.status_code == 200 else []
    ok = r.status_code == 200 and len(sentences) > 0
    check(
        "פתיחת מסך שיעור",
        "מסך השיעור נפתח ומוכן להתחלת הלמידה",
        f"פרטי שיעור הוחזרו, {len(sentences)} משפטי תרגול",
        ok,
    )

    ordered = [s["order_index"] for s in sentences]
    check(
        "משפטי התרגול מוחזרים בסדר הנכון",
        "המשפטים מסודרים לפי סדר הופעתם בשיעור",
        f"סדר: {ordered}",
        ordered == sorted(ordered),
    )

    r = httpx.get(f"{base}/categories/does-not-exist", headers=auth(token), timeout=20)
    check(
        "בחירת קטגוריה שאינה קיימת",
        "המערכת מחזירה שגיאה ואינה קורסת",
        f"HTTP {r.status_code}",
        r.status_code in (404, 422),
    )
    return category_id, lesson_id


# ---------------------------------------------------------------- 10.1.3
def test_resume(base: str, token: str, lesson_id: str) -> None:
    print("\n== 10.1.3 : חידוש שיעור שלא הושלם ==")
    r = httpx.post(f"{base}/lessons/{lesson_id}/start", headers=auth(token), timeout=20)
    run_a = r.json().get("run_id") if r.status_code == 200 else None
    check(
        "התחלת שיעור חדש",
        "נוצר מזהה ריצה חדש עבור השיעור",
        f"run_id = {run_a}",
        bool(run_a),
    )

    r = httpx.post(
        f"{base}/lessons/{lesson_id}/start",
        headers=auth(token), params={"is_resume": "true"}, timeout=20,
    )
    run_b = r.json().get("run_id") if r.status_code == 200 else None
    check(
        "בחירת המשך שיעור",
        "המערכת מזהה התקדמות שמורה וממשיכה את אותה ריצה",
        "אותו מזהה ריצה" if run_b == run_a else "מזהה ריצה שונה",
        run_b == run_a and run_b is not None,
    )

    r = httpx.post(
        f"{base}/lessons/{lesson_id}/start",
        headers=auth(token), params={"is_resume": "false"}, timeout=20,
    )
    run_c = r.json().get("run_id") if r.status_code == 200 else None
    check(
        "בחירת התחלה מחדש",
        "ההתקדמות הקודמת מתאפסת ונוצרת ריצה חדשה",
        "מזהה ריצה חדש" if run_c != run_a else "אותו מזהה ריצה",
        run_c is not None and run_c != run_a,
    )

    r = httpx.get(f"{base}/lessons/{lesson_id}", headers=auth(token), timeout=20)
    d = r.json() if r.status_code == 200 else {}
    check(
        "איפוס מונה ההתקדמות לאחר התחלה מחדש",
        "מספר המשפטים שהושלמו בריצה החדשה הוא 0",
        f"completed = {d.get('completed_sentences')} מתוך {d.get('sentences_count')}",
        d.get("completed_sentences") == 0,
    )

    r = httpx.post(
        f"{base}/lessons/{lesson_id}/complete",
        headers=auth(token), json={"run_id": "00000000-0000-0000-0000-000000000000"}, timeout=20,
    )
    check(
        "סיום שיעור עם מזהה ריצה שאינו תקף",
        "הבקשה נדחית ואינה משנה את ההתקדמות",
        f"HTTP {r.status_code}",
        r.status_code == 400,
    )


# ---------------------------------------------------------------- 10.1.4
def test_progress(base: str, token: str) -> None:
    print("\n== 10.1.4 : צפייה בהישגים ובהתקדמות הלמידה ==")
    r = httpx.get(base + "/progress/overview", headers=auth(token), timeout=20)
    check(
        "כניסה למסך התקדמות",
        "מסך ההתקדמות נטען והנתונים מוצגים",
        f"HTTP {r.status_code}, שדות: {list(r.json().keys()) if r.status_code == 200 else '-'}",
        r.status_code == 200,
    )

    r = httpx.get(base + "/progress/categories", headers=auth(token), timeout=20)
    cats = r.json() if r.status_code == 200 else []
    check(
        "צפייה בהתקדמות לפי קטגוריות",
        "מוצגות הקטגוריות עם נתוני התקדמות וציונים",
        f"{len(cats)} קטגוריות עם נתוני התקדמות",
        r.status_code == 200 and len(cats) > 0,
    )

    r = httpx.get(base + "/progress/badges", headers=auth(token), timeout=20)
    badges = r.json() if r.status_code == 200 else []
    described = all(b.get("description") for b in badges)
    check(
        "צפייה בתגים שנצברו",
        "מוצגים כל התגים האפשריים, כל אחד עם תיאור תנאי הזכייה",
        f"{len(badges)} תגים, לכולם תיאור" if described else f"{len(badges)} תגים, חלקם ללא תיאור",
        r.status_code == 200 and len(badges) > 0 and described,
    )

    r = httpx.get(base + "/progress/badges/unseen", headers=auth(token), timeout=20)
    check(
        "שליפת תגים שטרם הוצגו",
        "המערכת מחזירה את רשימת התגים שטרם הוצגו למשתמש",
        f"HTTP {r.status_code}, {len(r.json()) if r.status_code == 200 else '-'} תגים",
        r.status_code == 200,
    )


NOT_COVERED = [
    ("10.1.1", "אין קטגוריות זמינות / קטגוריה ללא שיעורים — דורש מצב מסד ייעודי"),
    ("10.1.2", "כל תסריטי ביצוע השיעור — דורשים מכשיר: הקראה, הקלטה, סנכרון שפתיים, משוב"),
    ("10.1.4", "מצב ללא נתוני התקדמות — דורש משתמש חדש"),
    ("10.1.5", "הרשמה, סיסמאות לא תואמות, דוא\"ל תפוס, התחברות Google — דורשים מסך"),
    ("10.1.6", "כל תסריטי ניהול הפרופיל והחשבון — דורשים מסך"),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8000")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    args = p.parse_args()
    base = args.base.rstrip("/")

    print(f"Backend: {base}")
    try:
        token = sign_in(args.email, args.password)
    except Exception as exc:
        print(f"Sign-in failed: {exc}")
        return 2
    print("Signed in, ID token obtained.\n")

    started = time.time()
    test_authentication(base, token)
    _, lesson_id = test_category_and_lesson_selection(base, token)
    test_resume(base, token, lesson_id)
    test_progress(base, token)
    elapsed = time.time() - started

    passed = sum(1 for *_, ok in results if ok)
    print("\n" + "=" * 72)
    print(f"  {passed}/{len(results)} scenarios PASSED   ({elapsed:.1f}s)")
    print("=" * 72)
    for scenario, _, actual, ok in results:
        if not ok:
            print(f"  FAIL  {scenario}  ->  {actual}")

    print("\nNOT COVERED by this script (must be verified manually on a device):")
    for section, what in NOT_COVERED:
        print(f"  {section}  {what}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
