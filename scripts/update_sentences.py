"""One-shot DB migration: improve pedagogical quality of practice sentences.

Each (old, new) pair updates by exact text match. Rows keep their IDs so
UserSentenceProgress and other FKs are untouched. Runs in a single
transaction — partial failure rolls back, leaving the DB unchanged.

Run from the repo root:
    python scripts/update_sentences.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.database import SessionLocal
from app.models.domain import Sentence

# (old_text, new_text). Old text must match the DB exactly.
UPDATES: list[tuple[str, str]] = [
    # --- Dining & Ordering Food ---
    ("What ingredients are in this", "What ingredients are in this dish?"),
    ("Is this dish spicy", "Is this dish spicy?"),
    ("Do you have vegetarian options", "Do you have any vegetarian options?"),
    ("What's the portion size", "How big are the portions?"),
    ("Does this contain any gluten traces", "Does this contain any gluten?"),
    ("I am lactose intolerant, need alternatives", "Do you have any lactose-free options?"),
    ("Please avoid cross contamination with nuts", "Please make sure there's no cross-contamination with nuts."),
    ("Can I schedule a pickup", "Can I schedule a pickup time?"),
    ("Extra sauce on the side", "Can I have extra sauce on the side?"),
    ("Please make it well cooked", "Please cook it well done."),
    ("Can I see the menu", "Could I see the menu?"),
    ("Please make it mild", "Could you make it mild?"),
    ("Can we split the bill evenly", "Can we split the bill evenly?"),
    ("Is service charge included in total", "Is the service charge included in the total?"),
    ("Authorize payment via contactless card", "I'd like to pay by card."),
    ("Please provide a detailed invoice", "Could I get an itemized receipt?"),

    # --- Emergencies & Safety ---
    ("I need an ambulance please", "I need an ambulance immediately."),
    ("Call the police now please", "Please call the police right away."),
    ("There is a fire here", "There's a fire in the building!"),
    ("Someone is injured here", "Someone has been seriously injured."),
    ("Call my emergency contact now", "Please call my emergency contact."),
    ("Here is my contact number", "Here is my phone number."),
    ("Do not use the elevator, use stairs", "Use the stairs, not the elevator."),
    ("There is heavy smoke, stay low", "Stay low because of the smoke."),
    ("Report fire to building management immediately", "Please report the fire to building management immediately."),
    ("How do I report theft?", "How do I report a theft?"),
    ("I am having severe chest pain", "I'm having severe chest pain."),
    ("I am allergic to penicillin", "I'm allergic to penicillin."),
    ("He is unconscious and not breathing", "He's unconscious and not breathing."),
    ("I swallowed something toxic, help", "I've swallowed something poisonous."),

    # --- Idioms & Everyday Expressions ---
    ("Think outside the box for solutions", "We need to think outside the box."),
    ("Hit the ground running on projects", "Let's hit the ground running."),
    ("Bite off more than you can chew", "Don't bite off more than you can chew."),
    ("Keep me posted on that", "Please keep me posted."),
    ("That makes total sense now", "That makes total sense."),
    ("I'm really surprised", "I'm really surprised!"),
    ("That's amazing, well done", "That's amazing work!"),
    ("Wow, that's great news", "Wow, that's great news!"),
    ("That's so disappointing", "That's really disappointing."),

    # --- Job Interview ---
    ("I managed a small team", "I managed a small team of engineers."),
    ("I seek a competitive market salary", "I'm looking for a competitive salary."),
    ("Can we negotiate stock options and bonus?", "Can we negotiate stock options and bonuses?"),
    ("I would prefer performance based incentives", "I'd prefer performance-based incentives."),
    ("I am a software developer", "I'm a software developer."),
    ("I have three years experience", "I have three years of experience."),

    # --- Public Transport ---
    ("I require wheelchair access, please assist", "Could you help me with wheelchair access?"),
    ("Is there priority seating for disabled passengers", "Is there priority seating for disabled passengers?"),
    ("Can staff provide visual assistance, please", "Could a staff member help guide me?"),
    ("Where is the accessible entrance located", "Where is the accessible entrance?"),
    ("How do I get there", "How do I get there?"),
    ("Which stop for the museum", "Which stop should I take for the museum?"),
    ("Can you show me on map", "Could you show me on the map?"),
    ("Is it within walking distance", "Is it within walking distance?"),
    ("My train is significantly delayed", "My train is running very late."),
    ("How can I claim compensation", "How can I claim compensation?"),
    ("Please explain the delay reason", "Could you explain the reason for the delay?"),
    ("When is the next bus", "When is the next bus?"),
    ("Which line goes downtown", "Which line goes downtown?"),
    ("What time does it depart", "What time does the bus depart?"),
    ("Is there a night service", "Is there a night service?"),
    ("Please observe priority seating rules", "Please respect the priority seating."),
    ("Keep noise to a minimum please", "Please keep the noise down."),
    ("Offer your seat to elderly passengers", "Please offer your seat to elderly passengers."),
    ("How much is a ticket", "How much is a ticket?"),
    ("Single or return ticket?", "I'd like a return ticket, please."),
    ("Where can I buy tickets", "Where can I buy tickets?"),
    ("Is contactless payment accepted", "Do you accept contactless payment?"),
    ("How long is the transfer time", "How long is the transfer?"),
    ("Which platform for the connection", "Which platform is my connection?"),
    ("Is there a direct route", "Is there a direct route?"),

    # --- Small Talk & Socializing ---
    ("You did a great job", "You did a great job!"),
    ("That outfit looks really good", "That outfit looks great on you!"),
    ("Thank you, that's very kind", "Thank you — that's very kind of you."),
    ("I appreciate your help", "I really appreciate your help."),
    ("I collect vintage vinyl records at home", "I collect vintage vinyl records."),
    ("I practice guitar for two hours", "I practice guitar for two hours every day."),
    ("What do you do?", "What do you do for work?"),
    ("It's a sunny day", "It's a beautiful sunny day."),
    ("I have a long day", "I have a long day ahead of me."),

    # --- Tech Support ---
    ("Reset my password please", "Could you reset my password?"),
    ("Enable two factor authentication", "Please enable two-factor authentication."),
    ("I cannot access my account", "I can't access my account."),
    ("Please share your screen via remote session", "Could you share your screen with me?"),
    ("Grant temporary administrative privileges for diagnostics", "I need temporary admin access to run diagnostics."),
    ("Run the diagnostic tool and send logs", "Please run the diagnostic tool and send me the logs."),
    ("Authorize remote control to replicate issue", "Please allow remote access so I can reproduce the issue."),
    ("Check system requirements before installation", "Please check the system requirements before installing."),
    ("The update failed with error code", "The update failed and gave me an error code."),
    ("Please run installer as administrator", "Please run the installer as administrator."),
    ("Describe the problem briefly", "Could you describe the problem briefly?"),
    ("Have you tried restarting?", "Have you tried restarting it?"),
    ("Check the device power", "Make sure the device is plugged in."),

    # --- Travel Abroad ---
    ("Where is the check in desk?", "Where is the check-in desk?"),
    ("I need to drop my luggage", "I need to drop off my luggage."),
    ("Can I get a late checkout please?", "Could I have a late checkout?"),
    ("My room has a maintenance issue", "There's a problem with my room."),
    ("Is breakfast included in the rate?", "Is breakfast included in the price?"),
    ("Is the fare refundable or nonrefundable?", "Is this ticket refundable?"),
    ("Please confirm my booking reference number", "Could you confirm my booking reference?"),
    ("I request a full refund policy explanation", "Could you explain the refund policy?"),
    ("Which form must I fill?", "Which form do I need to fill out?"),
    ("Do you recommend local tours?", "Could you recommend a local tour?"),
]


def main() -> None:
    updated_count = 0
    not_found: list[str] = []
    already_done: list[str] = []

    with SessionLocal() as db:
        try:
            for old, new in UPDATES:
                rows = db.query(Sentence).filter(Sentence.text == old).all()
                if not rows:
                    if db.query(Sentence).filter(Sentence.text == new).first():
                        already_done.append(old)
                    else:
                        not_found.append(old)
                    continue
                for row in rows:
                    row.text = new
                updated_count += len(rows)
            db.commit()
        except Exception:
            db.rollback()
            raise

    total = len(UPDATES)
    print(f"\n=== Sentence migration summary ===")
    print(f"Updates attempted : {total}")
    print(f"Rows updated      : {updated_count}")
    print(f"Already up-to-date: {len(already_done)}")
    print(f"Not found in DB   : {len(not_found)}")

    if already_done:
        print(f"\nAlready up-to-date ({len(already_done)}):")
        for s in already_done:
            print(f"  - {s!r}")
    if not_found:
        print(f"\nNot found ({len(not_found)}) - check the OLD text matches DB exactly:")
        for s in not_found:
            print(f"  - {s!r}")


if __name__ == "__main__":
    main()
