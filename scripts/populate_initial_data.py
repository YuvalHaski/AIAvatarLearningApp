import uuid
from app.core.database import SessionLocal
from app.models.domain import Category, Lesson, Sentence, DifficultyEnum, Badge

# ==========================================
# 1. INITIAL DATA STRUCTURE
# ==========================================
# This list contains the core content for the app.
# Each dictionary represents a Category, containing a list of Lessons,
# which in turn contain a list of Sentences.

INITIAL_DATA = [
    {
        "title": "Job Interview",
        "description": "Practical interview phrases to help you start conversations, summarize your work experience, and discuss salary and benefits with confidence.",
        "icon": "",
        "lessons": [
            {
                "title": "Self Introduction",
                "difficulty": DifficultyEnum.EASY,
                "description": "Natural opening lines to introduce yourself.",
                "sentences": [
                    "Hello, my name is Israel",
                    "I am a software developer",
                    "Nice to meet you",
                    "I have three years experience"
                ]
            },
            {
                "title": "Discussing Experience",
                "difficulty": DifficultyEnum.MEDIUM,
                "description": "Describe roles, achievements, and technical responsibilities clearly.",
                "sentences": [
                    "I managed a small team",
                    "I improved app performance by 30%",
                    "I developed REST APIs with Spring",
                    "I optimized database queries for speed"
                ]
            },
            {
                "title": "Salary & Negotiation",
                "difficulty": DifficultyEnum.HARD,
                "description": "Phrases to discuss salary, benefits, and negotiation.",
                "sentences": [
                    "What is the total compensation package?",
                    "I seek a competitive market salary",
                    "Can we negotiate stock options and bonus?",
                    "I would prefer performance based incentives"
                ]
            }
        ]
    },
    {
        "title": "Travel Abroad",
        "description": "Key travel phrases for airports, hotels, bookings, and getting around, so travelers can handle check-in, customs, reservations, and directions with confidence.",
        "icon": "",
        "lessons": [
            {
                "title": "Airport Procedures",
                "difficulty": DifficultyEnum.MEDIUM,
                "description": "Essential phrases for check-in, security, boarding, and baggage handling.",
                "sentences": [
                    "Where is the check in desk?",
                    "I need to drop my luggage",
                    "What time does boarding start?",
                    "Where is the departure gate?"
                ]
            },
            {
                "title": "Sightseeing & Directions",
                "difficulty": DifficultyEnum.EASY,
                "description": "Simple phrases for asking directions, finding attractions, and local recommendations.",
                "sentences": [
                    "How do I get there?",
                    "Is it within walking distance?",
                    "Where is the nearest museum?",
                    "Do you recommend local tours?"
                ]
            },
            {
                "title": "Customs & Immigration",
                "difficulty": DifficultyEnum.EASY,
                "description": "Clear lines for passport control, declarations, and visa questions.",
                "sentences": [
                    "Where is passport control?",
                    "I have nothing to declare",
                    "Do I need a visa?",
                    "Which form must I fill?"
                ]
            },
            {
                "title": "Booking & Reservations",
                "difficulty": DifficultyEnum.HARD,
                "description": "Formal phrases for changing bookings, refund policies, and confirming references.",
                "sentences": [
                    "Can I change my reservation date?",
                    "Is the fare refundable or nonrefundable?",
                    "Please confirm my booking reference number",
                    "I request a full refund policy explanation"
                ]
            },
            {
                "title": "At the Hotel",
                "difficulty": DifficultyEnum.HARD,
                "description": "Useful hotel phrases for check-in, room requests, maintenance issues, and billing.",
                "sentences": [
                    "I have a reservation under my name",
                    "Can I get a late checkout please?",
                    "My room has a maintenance issue",
                    "Is breakfast included in the rate?"
                ]
            }
        ]
    },
    {
        "title": "Emergencies & Safety",
        "description": "Essential, easy-to-use phrases for urgent situations: call contacts and emergency services, describe medical problems, report theft, and follow fire evacuation instructions.",
        "icon": "",
        "lessons": [
            {
                "title": "Emergency Contacts",
                "difficulty": DifficultyEnum.EASY,
                "description": "Quick lines to share emergency contact details and request a call.",
                "sentences": [
                    "Call my emergency contact now",
                    "Here is my contact number",
                    "My emergency contact is Angela"
                ]
            },
            {
                "title": "Calling Emergency Services",
                "difficulty": DifficultyEnum.EASY,
                "description": "Short, direct phrases to request police, ambulance, or fire assistance.",
                "sentences": [
                    "I need an ambulance please",
                    "Call the police now please",
                    "There is a fire here",
                    "Someone is injured here"
                ]
            },
            {
                "title": "Medical Emergencies",
                "difficulty": DifficultyEnum.MEDIUM,
                "description": "Clear phrases to describe symptoms, allergies, and life-threatening conditions.",
                "sentences": [
                    "I am having severe chest pain",
                    "I am allergic to penicillin",
                    "He is unconscious and not breathing",
                    "I swallowed something toxic, help"
                ]
            },
            {
                "title": "Lost or Stolen Items",
                "difficulty": DifficultyEnum.MEDIUM,
                "description": "Useful lines to report lost or stolen property and find the lost-and-found.",
                "sentences": [
                    "My passport was stolen yesterday",
                    "I lost my wallet today",
                    "Where is the lost and found?",
                    "How do I report theft?"
                ]
            },
            {
                "title": "Fire & Evacuation",
                "difficulty": DifficultyEnum.HARD,
                "description": "Clear instructions for evacuations, smoke safety, and reporting fires to authorities.",
                "sentences": [
                    "Follow evacuation signs to the assembly point",
                    "Do not use the elevator, use stairs",
                    "There is heavy smoke, stay low",
                    "Report fire to building management immediately"
                ]
            }
        ]
    },
    {
        "title": "Small Talk & Socializing",
        "description": "Friendly phrases for meeting people, chatting about everyday life, and making casual plans.",
        "icon": "",
        "lessons": [
            {
                "title": "Introductions",
                "difficulty": DifficultyEnum.EASY,
                "description": "Simple lines to exchange names, jobs, and quick background.",
                "sentences": [
                    "Hi, I'm Angela",
                    "Nice to meet you",
                    "Where are you from?",
                    "What do you do?"
                ]
            },
            {
                "title": "Weather & Daily Life",
                "difficulty": DifficultyEnum.EASY,
                "description": "Casual phrases for talking about the weather, routines, and how your day is going.",
                "sentences": [
                    "How's the weather today?",
                    "It's a sunny day",
                    "Did you sleep well?",
                    "I have a long day"
                ]
            },
            {
                "title": "Hobbies & Interests",
                "difficulty": DifficultyEnum.HARD,
                "description": "Phrases to describe hobbies and ask about other people's interests in more detail.",
                "sentences": [
                    "I enjoy painting landscapes on weekends",
                    "I collect vintage vinyl records at home",
                    "I practice guitar for two hours",
                    "Do you play any team sports?"
                ]
            },
            {
                "title": "Compliments & Responses",
                "difficulty": DifficultyEnum.MEDIUM,
                "description": "How to give polite compliments and respond graciously.",
                "sentences": [
                    "You did a great job",
                    "That outfit looks really good",
                    "Thank you, that's very kind",
                    "I appreciate your help"
                ]
            },
            {
                "title": "Invitations & Plans",
                "difficulty": DifficultyEnum.HARD,
                "description": "Friendly ways to invite, confirm, reschedule, and set meeting times.",
                "sentences": [
                    "Would you like to grab coffee tomorrow?",
                    "Let's meet after work on Friday",
                    "Can we reschedule to next week?",
                    "I'm free Wednesday evening after six"
                ]
            }
        ]
    },
    {
        "title": "Dining & Ordering Food",
        "description": "Common restaurant and food phrases for ordering, asking about dishes, handling dietary needs, takeout, and paying; useful in cafes, restaurants, and food stalls.",
        "icon": "",
        "lessons": [
            {
                "title": "Ordering at a Restaurant",
                "difficulty": DifficultyEnum.EASY,
                "description": "Simple phrases to order meals, request the menu, and specify preferences.",
                "sentences": [
                    "I'd like the chicken salad",
                    "Can I see the menu",
                    "I'll have the soup",
                    "Please make it mild"
                ]
            },
            {
                "title": "Asking About the Menu",
                "difficulty": DifficultyEnum.EASY,
                "description": "Handy questions to check ingredients, spice level, and portion size.",
                "sentences": [
                    "What ingredients are in this",
                    "Is this dish spicy",
                    "Do you have vegetarian options",
                    "What's the portion size"
                ]
            },
            {
                "title": "Dietary Restrictions",
                "difficulty": DifficultyEnum.HARD,
                "description": "Clear lines to state allergies or intolerances, and request safe alternatives.",
                "sentences": [
                    "I'm allergic to tree nuts",
                    "Does this contain any gluten traces",
                    "I am lactose intolerant, need alternatives",
                    "Please avoid cross contamination with nuts"
                ]
            },
            {
                "title": "Fast Food & Takeaway",
                "difficulty": DifficultyEnum.MEDIUM,
                "description": "Useful phrases for ordering to-go, scheduling pickups, and adding special instructions.",
                "sentences": [
                    "I'd like this to go",
                    "Can I schedule a pickup",
                    "Extra sauce on the side",
                    "Please make it well cooked"
                ]
            },
            {
                "title": "Payment & Splitting the Bill",
                "difficulty": DifficultyEnum.HARD,
                "description": "Polite sentences for asking the check, splitting payments, and confirming charges or invoices.",
                "sentences": [
                    "Can we split the bill evenly",
                    "Is service charge included in total",
                    "Authorize payment via contactless card",
                    "Please provide a detailed invoice"
                ]
            }
        ]
    },
    {
        "title": "Idioms & Everyday Expressions",
        "description": "Useful idioms and everyday phrases to sound more natural, understand common expressions, and respond clearly in daily conversations.",
        "icon": "",
        "lessons": [
            {
                "title": "Common Idioms",
                "difficulty": DifficultyEnum.HARD,
                "description": "Common idioms used in conversation, learn their meanings and practical usage.",
                "sentences": [
                    "Break the ice with small talk",
                    "Think outside the box for solutions",
                    "Hit the ground running on projects",
                    "Bite off more than you can chew"
                ]
            },
            {
                "title": "Everyday Expressions",
                "difficulty": DifficultyEnum.MEDIUM,
                "description": "Short, everyday phrases for routine interactions and quick replies.",
                "sentences": [
                    "Long time no see",
                    "I'll be right back",
                    "Keep me posted on that",
                    "That makes total sense now"
                ]
            },
            {
                "title": "Reactions & Emotions",
                "difficulty": DifficultyEnum.EASY,
                "description": "Simple reactions to show surprise, joy, disappointment, or praise.",
                "sentences": [
                    "I'm really surprised",
                    "That's amazing, well done",
                    "Wow, that's great news",
                    "That's so disappointing"
                ]
            }
        ]
    },
    {
        "title": "Tech Support",
        "description": "Practical phrases to describe problems, manage accounts, install or update software, and perform remote diagnostics.",
        "icon": "",
        "lessons": [
            {
                "title": "Troubleshooting Basics",
                "difficulty": DifficultyEnum.EASY,
                "description": "Short phrases to help with issues and try basic fixes.",
                "sentences": [
                    "Describe the problem briefly",
                    "Have you tried restarting?",
                    "I can't reproduce the issue",
                    "Check the device power"
                ]
            },
            {
                "title": "Account & Passwords",
                "difficulty": DifficultyEnum.EASY,
                "description": "Clear lines for password resets, access problems, and security settings.",
                "sentences": [
                    "I forgot my password",
                    "Reset my password please",
                    "Enable two factor authentication",
                    "I cannot access my account"
                ]
            },
            {
                "title": "Software Installation & Updates",
                "difficulty": DifficultyEnum.MEDIUM,
                "description": "Useful phrases for installation steps, requirements, and update errors.",
                "sentences": [
                    "How do I install this?",
                    "Check system requirements before installation",
                    "The update failed with error code",
                    "Please run installer as administrator"
                ]
            },
            {
                "title": "Remote Support & Instructions",
                "difficulty": DifficultyEnum.HARD,
                "description": "Formal phrases to request screen sharing, permissions, diagnostics, and remote control.",
                "sentences": [
                    "Please share your screen via remote session",
                    "Grant temporary administrative privileges for diagnostics",
                    "Run the diagnostic tool and send logs",
                    "Authorize remote control to replicate issue"
                ]
            }
        ]
    },
    {
        "title": "Public Transport",
        "description": "Practical phrases for using buses, trains and metros: buy tickets, check routes and schedules, ask for directions or assistance, and handle delays or safety issues.",
        "icon": "",
        "lessons": [
            {
                "title": "Ticketing & Fares",
                "difficulty": DifficultyEnum.EASY,
                "description": "Short questions to buy tickets, check fares, and confirm payment methods.",
                "sentences": [
                    "How much is a ticket",
                    "Single or return ticket?",
                    "Where can I buy tickets",
                    "Is contactless payment accepted"
                ]
            },
            {
                "title": "Routes & Schedules",
                "difficulty": DifficultyEnum.EASY,
                "description": "Quick lines to ask about next departures, lines, and service times.",
                "sentences": [
                    "When is the next bus",
                    "Which line goes downtown",
                    "What time does it depart",
                    "Is there a night service"
                ]
            },
            {
                "title": "Asking Directions",
                "difficulty": DifficultyEnum.MEDIUM,
                "description": "Phrases to ask which stop to use, request map help, and confirm walking distance.",
                "sentences": [
                    "How do I get there",
                    "Which stop for the museum",
                    "Can you show me on map",
                    "Is it within walking distance"
                ]
            },
            {
                "title": "Transfers & Connections",
                "difficulty": DifficultyEnum.MEDIUM,
                "description": "Useful lines for changing lines, platform information, and transfer times.",
                "sentences": [
                    "How long is the transfer time",
                    "Which platform for the connection",
                    "Is there a direct route"
                ]
            },
            {
                "title": "Accessibility & Assistance",
                "difficulty": DifficultyEnum.HARD,
                "description": "Polite requests for wheelchair access, priority seating, and staff assistance.",
                "sentences": [
                    "I require wheelchair access, please assist",
                    "Is there priority seating for disabled passengers",
                    "Can staff provide visual assistance, please",
                    "Where is the accessible entrance located"
                ]
            },
            {
                "title": "Delays & Complaints",
                "difficulty": DifficultyEnum.HARD,
                "description": "Formal phrases to report delays, ask about compensation, and file complaints.",
                "sentences": [
                    "My train is significantly delayed",
                    "How can I claim compensation",
                    "Please explain the delay reason",
                    "I need to file a formal complaint"
                ]
            },
            {
                "title": "Safety & Etiquette",
                "difficulty": DifficultyEnum.HARD,
                "description": "Polite reminders and rules for safe, considerate travel on public transport.",
                "sentences": [
                    "Please observe priority seating rules",
                    "Keep noise to a minimum please",
                    "Do not block emergency exits",
                    "Offer your seat to elderly passengers"
                ]
            }
        ]
    }
]

# ==========================================
# 2. INITIAL BADGES STRUCTURE
# ==========================================
# We use gaps of 10 in the order_index to allow inserting new badges in the future
# without needing to re-index all existing badges in the database.

INITIAL_BADGES = [
    {
        "title": "First Step",
        "description": "Completed your very first lesson.",
        "icon": "",
        "order_index": 10
    },
    {
        "title": "Perfect Score",
        "description": "Completed a lesson with 100% accuracy.",
        "icon": "",
        "order_index": 20
    },
    {
        "title": "Conversation Starter",
        "description": "Complete 5 Lessons.",
        "icon": "",
        "order_index": 30
    },
    {
        "title": "Lesson Collector",
        "description": "Finish 10 different lessons.",
        "icon": "",
        "order_index": 40
    },
    {
        "title": "7 Day Streak",
        "description": "Practice at least one lesson for 7 consecutive days.",
        "icon": "",
        "order_index": 50
    },
    {
        "title": "Early Bird",
        "description": "Practice between 6 AM and 8 AM once.",
        "icon": "",
        "order_index": 60
    },
    {
        "title": "Night Owl",
        "description": "Practice between midnight and 4 AM once.",
        "icon": "",
        "order_index": 70
    },
    {
        "title": "Traveler",
        "description": "Complete all lessons in Travel Abroad category.",
        "icon": "",
        "order_index": 80
    },
    {
        "title": "Foodie",
        "description": "Complete all Dining & Ordering Food lessons.",
        "icon": "",
        "order_index": 90
    },
    {
        "title": "Transport Expert",
        "description": "Complete all Public Transport lessons.",
        "icon": "",
        "order_index": 100
    },
    {
        "title": "Techie",
        "description": "Complete all Tech Support lessons.",
        "icon": "",
        "order_index": 110
    },
    {
        "title": "Social Butterfly",
        "description": "Complete all Small Talk & Socializing lessons.",
        "icon": "",
        "order_index": 120
    },
    {
        "title": "Badge Collector",
        "description": "Earn 10 different badges.",
        "icon": "",
        "order_index": 130
    },
    {
        "title": "30 Day Streak",
        "description": "Practice at least one lesson for 30 consecutive days.",
        "icon": "",
        "order_index": 140
    },
    {
        "title": "Mastery",
        "description": "Complete every lesson in the app.",
        "icon": "",
        "order_index": 150
    }
]


def seed_database():
    """
    Connects to the database and populates it with the initial Categories,
    Lessons, Sentences, and Badges if they don't already exist.
    """
    db = SessionLocal()

    try:
        # Check if we already have Categories data
        if not db.query(Category).first():
            print("Starting to seed database with Categories, Lessons, and Sentences...")

            for cat_data in INITIAL_DATA:
                # 1. Create the Category
                category_id = str(uuid.uuid4())
                new_category = Category(
                    id=category_id,
                    title=cat_data["title"],
                    description=cat_data["description"],
                    icon=cat_data["icon"]
                )
                db.add(new_category)

                # 2. Iterate through and create Lessons for this Category
                for lesson_data in cat_data["lessons"]:
                    lesson_id = str(uuid.uuid4())
                    new_lesson = Lesson(
                        id=lesson_id,
                        category_id=category_id,
                        title=lesson_data["title"],
                        description=lesson_data["description"],
                        difficulty=lesson_data["difficulty"],
                        icon=""
                    )
                    db.add(new_lesson)

                    # 3. Iterate through and create Sentences for this Lesson
                    for index, sentence_text in enumerate(lesson_data["sentences"], start=1):
                        new_sentence = Sentence(
                            id=str(uuid.uuid4()),
                            lesson_id=lesson_id,
                            text=sentence_text,
                            order_index=index
                        )
                        db.add(new_sentence)
        else:
            print("Categories already exist in the database. Skipping Categories seeding...")

        # Check separately for Badges in case they were added later
        if not db.query(Badge).first():
            print("Seeding Badges...")
            for badge_data in INITIAL_BADGES:
                new_badge = Badge(
                    id=str(uuid.uuid4()),
                    title=badge_data["title"],
                    description=badge_data["description"],
                    icon=badge_data["icon"],
                    order_index=badge_data["order_index"]
                )
                db.add(new_badge)
        else:
            print("Badges already exist in the database. Skipping Badges seeding...")

        # Commit all the transactions to the database
        db.commit()
        print("Database seeding process finished successfully! 🎉")

    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()