# AI Avatar Learning App — Backend

Backend for a "repeat after me" English pronunciation trainer. A 3D avatar speaks a sentence,
the learner repeats it, and the system returns a numeric score plus spoken, specific coaching
feedback — *which* word was wrong, *which* sound inside it, and how to fix it.

This repository contains the **FastAPI backend**, the **deterministic pronunciation analyzer**,
and the **fine-tuning pipeline** for the feedback model.
The Android client lives in a separate repository: **[Lioravraham5/LearningApp](https://github.com/Lioravraham5/LearningApp)**.

---

## Table of Contents

1. [What the system does](#1-what-the-system-does)
2. [Architecture](#2-architecture)
3. [The feedback pipeline](#3-the-feedback-pipeline)
4. [Key code — links and descriptions](#4-key-code--links-and-descriptions)
5. [Repository layout](#5-repository-layout)
6. [Getting started](#6-getting-started)
7. [API reference](#7-api-reference)
8. [Data model](#8-data-model)
9. [Testing](#9-testing)
10. [Model training pipeline](#10-model-training-pipeline)
11. [Known limitations](#11-known-limitations)

---

## 1. What the system does

| Capability | Description |
| --- | --- |
| Catalog | Categories → lessons → sentences, seeded into PostgreSQL. |
| Authentication | Firebase ID tokens verified server-side; users mirrored into the local DB. |
| Pronunciation assessment | Azure Speech *scripted* Pronunciation Assessment returns accuracy / fluency / completeness / prosody plus per-word and per-phoneme scores. |
| Deterministic analysis | Our own code decides what went wrong (missing / extra / substituted / mispronounced words), computes the final 0–100 score, and picks what is worth saying. |
| Spoken feedback | A fine-tuned small LLM turns that structured analysis into 2–5 tutor-style sentences the avatar reads aloud. A template renderer is the always-available fallback. |
| Progress & badges | Per-run attempt history, lesson completion, average scores, and a rule-based badge engine. |

**Design principle:** the model never decides *what is true*. Every judgement (score, pass/fail,
which sound was wrong) is made by deterministic Python; the model only performs *wording*. This is
what keeps feedback trustworthy and testable.

---

## 2. Architecture

```mermaid
flowchart LR
    User["Learner"] --> Android["Android App<br/>Kotlin · Jetpack Compose"]
    Android --> FBAuth["Firebase Auth SDK"]
    Android --> TTS["Azure TTS<br/>avatar speech + visemes"]
    Android -- "audio + sentence_id" --> API["FastAPI Backend<br/>(this repo)"]

    API --> Admin["Firebase Admin SDK<br/>token verification"]
    API --> DB[("PostgreSQL<br/>SQLAlchemy + Alembic")]
    API --> Azure["Azure Speech<br/>Pronunciation Assessment"]
    API --> LLM["Fine-tuned feedback model<br/>OpenAI-compatible endpoint"]
    API -- "score + feedback_text" --> Android
```

**Stack:** FastAPI · SQLAlchemy · Alembic · PostgreSQL · Firebase Admin · Azure Cognitive Services
Speech · `imageio-ffmpeg` · httpx · pytest. Training: PyTorch · PEFT/QLoRA · TRL · Transformers.

---

## 3. The feedback pipeline

A single `POST /asr` call runs the whole chain:

```mermaid
sequenceDiagram
    participant App as Android app
    participant API as POST /asr
    participant DB as PostgreSQL
    participant AZ as Azure Assessment
    participant S1 as Stage 1 — Analyzer
    participant S2 as Stage 2 — Feedback model

    App->>API: audio (m4a) + sentence_id + run_id
    API->>DB: resolve canonical sentence text
    API->>API: normalize audio → 16 kHz mono PCM WAV
    API->>AZ: assess against reference text
    AZ-->>API: recognized text, 4 sub-scores, per-word/per-phoneme detail
    API->>S1: analyze(target, asr_result)
    S1-->>API: ErrorReport (errors + final score + feedback points)
    API->>S2: generate_feedback(report)
    S2-->>API: spoken feedback text (verified for coverage)
    API->>DB: record attempt, update lesson progress
    API-->>App: AssessmentResponse
```

### Stage 1 — deterministic analyzer

Independent of Azure's own miscue flags, the target sentence and the recognized text are aligned
word-by-word with edit-distance alignment, producing exact `missing` / `extra` / `substitution`
operations. Phoneme evidence (including Azure's *NBestPhonemes* — what Azure believes was actually
produced) then decides which words are genuinely mispronounced and which sound to coach.

**Final score formula** ([`scoring.py`](app/services/feedback/scoring.py#L15-L48)):

```
base    = 0.40·accuracy + 0.30·completeness + 0.20·fluency + 0.10·prosody
penalty = 4·(substitutions) + 2·(extra words)
score   = round( clamp( base − penalty, 0, 100 ) )
passed  = score ≥ 80
```

Missing words are deliberately **not** penalized twice — they already depress Azure's
`completeness` sub-score.

### Stage 2 — wording

The `ErrorReport` is serialized to compact JSON and sent to a QLoRA fine-tuned Phi-3.5-mini behind
an OpenAI-compatible endpoint (Ollama locally, vLLM on a GPU host). Because the same
`build_model_messages` function is used at training time and at request time, the model is always
served exactly the prompt format it was trained on.

Two safety nets guarantee the learner never loses information because a language model got creative:

* **Coverage verifier** — if the generated text fails to name a concrete mistake (or omits a
  required hint verbatim), template sentences are appended, or the whole reply is replaced.
* **Template fallback** — if the model endpoint is unset, slow, or failing, the deterministic
  renderer produces the feedback instead. The endpoint never fails because of the model.

---

## 4. Key code — links and descriptions

The central algorithms, formulas and rule engines of the project, with what each one does and why
it matters.

### 4.1 Core algorithms and formulas

| Code | Role and importance |
| --- | --- |
| [`alignment.py` → `align()`](app/services/feedback/alignment.py#L32-L80) | **Needleman–Wunsch / Levenshtein edit-distance alignment** over word tokens (`O(n·m)` dynamic programming + backtracking). Produces the exact list of `match / substitution / missing / extra` operations between the target sentence and what was recognized. This is the ground truth for *what the learner actually said*, computed by us rather than trusted from the ASR's own flags. |
| [`scoring.py` → weights, penalties, `compute_score()`](app/services/feedback/scoring.py#L12-L52) | **The grading formula.** A weighted blend of Azure's four sub-scores (accuracy 0.40, completeness 0.30, fluency 0.20, prosody 0.10) minus per-mistake penalties, clamped to 0–100, with a pass threshold of 80. Every weight and threshold is a named constant so the rubric can be justified and tuned rather than hidden inside Azure's opaque aggregate. |
| [`analysis.py` → `analyze()`](app/services/feedback/analysis.py#L454-L523) | **Stage 1 entry point.** Orchestrates tokenization → alignment → mispronunciation detection → pattern detection → scoring → feedback-point selection, and returns the `ErrorReport` that everything downstream consumes. The single place where "what went wrong" is decided. |
| [`analysis.py` → `_weak_sound_pairs()`](app/services/feedback/analysis.py#L159-L203) | **Phoneme-level evidence filter.** Sorts a word's phonemes worst-first, drops sounds we cannot name to a learner (schwa, unanchored consonants), and uses Azure's NBest candidates conservatively: if Azure's best guess *is* the expected sound, the low score is treated as noise. This is what stops the avatar from coaching sounds the learner actually got right. |
| [`analysis.py` → `_confident_produced_phoneme()` + priority variant](app/services/feedback/analysis.py#L104-L156) | **Confidence gate for contrast hints.** Only names the sound the learner produced ("not the 'f' sound") when the top NBest candidate clears both a score floor and a margin over the expected phoneme, and belongs to the same vowel/consonant class. Prevents confident-sounding but wrong corrections. |
| [`analysis.py` → `_find_mispronounced()`](app/services/feedback/analysis.py#L269-L307) | **Three-tier mispronunciation detector.** Priority sounds (`th`, `dh`, `r`) get a more sensitive detector, Azure-confirmed mispronunciations get the standard threshold, and unflagged words need much stronger evidence before being surfaced. Encodes the product decision about *which* errors are worth a learner's attention. |
| [`analysis.py` → `_select_feedback_points()`](app/services/feedback/analysis.py#L357-L451) | **Prioritization and capping.** Assigns a priority to every issue, caps "framing" points at 3 and mispronunciations at 4 so the avatar stays short, and guarantees mispronunciations are never evicted by framing issues. Also emits a single *polish* tip explaining why a clean attempt still didn't reach 100. |
| [`analysis.py` → `_detect_patterns()`](app/services/feedback/analysis.py#L329-L339) | **Cross-word pattern detection.** A phoneme failing in ≥2 words becomes "recurring difficulty with the 'r' sound" — turns isolated errors into an actionable learning insight. |
| [`badge_rules.py` → streak rules, `_current_streak_length()`](app/services/badge_rules.py#L107-L149) | **Daily-streak algorithm.** Walks backwards over the set of distinct UTC practice dates, allowing today *or* yesterday as the anchor so an unextended streak isn't broken mid-day. Backs the 7-day and 30-day badges. |
| [`lesson_service.py` → `complete_lesson()`](app/services/lesson_service.py#L114-L189) | **Lesson score aggregation.** `MAX` score per sentence within the run, then `AVG` across sentences — so retries help and duplicate attempts can't distort the result. Also finalizes run state and triggers badge evaluation. |

### 4.2 Speech and audio layer

| Code | Role and importance |
| --- | --- |
| [`asr_service.py` → `assess_pronunciation()`](app/services/asr_service.py#L95-L192) | **Azure integration.** Runs *scripted* Pronunciation Assessment with the target sentence as reference text, phoneme granularity, miscue detection, prosody, and `nbest_phoneme_count=5`. Pins the phoneme alphabet to SAPI so returned codes match our hint tables — without it, voiced `th` can arrive as IPA and never gets coached. |
| [`asr_service.py` → `_nbest_candidates()`](app/services/asr_service.py#L38-L57) | Parses Azure's *NBestPhonemes* — its ranked guesses of the sound actually produced at each position. This is the raw evidence that makes "you said X instead of Y" possible instead of only "this was weak". |
| [`audio_utils.py` → `normalize_to_wav()`](app/services/audio_utils.py#L52-L110) | **Audio normalization via FFmpeg** to 16 kHz mono 16-bit PCM, with 0.3 s / 0.8 s silence padding so Azure can segment the first and last words cleanly. Writes the upload to a real temp file first, because M4A stores its index at the end and a non-seekable pipe silently yields an empty WAV — which Azure scores as a total omission. Includes a sanity check that real speech survived decoding. |

### 4.3 Coaching knowledge base

| Code | Role and importance |
| --- | --- |
| [`phoneme_hints.py` → anchor table](app/services/feedback/phoneme_hints.py#L80-L140) | **Phoneme → learner-friendly anchor word** ("the 'th' sound like in 'thin'"), for consonants and vowels, plus display graphemes so a raw SAPI code like `dh` is never shown. Schwa is deliberately absent — there is nothing meaningful to coach there. |
| [`phoneme_hints.py` → articulation lines](app/services/feedback/phoneme_hints.py#L149-L257) | **Articulation coaching** (mouth shape, tongue placement, airflow) for every anchored sound, so the avatar alternates between "sounds like in X" and real speech-therapist-style instruction. |
| [`phoneme_hints.py` → `contrast_hint_for()`](app/services/feedback/phoneme_hints.py#L378-L419) | Builds the contrast phrase *"the 'e' sound like in 'bed', not the 'a' sound"*, suppressing the contrast when both sounds display as the same grapheme. The hint is passed verbatim through Stage 2, so wording quality is controlled here, not by the model. |
| [`phoneme_hints.py` → `canonical_phoneme()`](app/services/feedback/phoneme_hints.py#L42-L52) | Normalizes Azure phoneme codes (strips stress digits, maps IPA → SAPI) so the rest of the system reasons about one alphabet. |
| [`word_hints.py` → `MINIMAL_PAIR_CONFUSABLES`](app/services/feedback/word_hints.py#L244-L337) | **Curated minimal pairs** (sheep/ship, right/light, wine/vine, think/sink…) mapped to the *one* sound that distinguishes them. When a real substitution matches a pair, the learner gets targeted sound coaching instead of "you said the wrong word". Covers the classic L1-interference sets: /iː–ɪ/, /æ–ɛ/, /uː–ʊ/, R/L, V/W, TH, Z/S. |
| [`word_hints.py` → silent letters, hard/soft C, clusters](app/services/feedback/word_hints.py#L52-L219) | **Orthography-driven coaching tables** for errors phoneme scoring cannot express: silent letters (`knife`, `comb`, `walk`), hard vs soft `c`, and difficult clusters (`str`, `thr`, `spr`). Paired with rule-level hints so the avatar teaches the rule, not only the instance. |
| [`word_hints.py` → `WORD_VOWEL_ANCHORS`](app/services/feedback/word_hints.py#L581-L654) | Word → dominant vowel phoneme. Consonant spelling triggers alone never produce vowel codes, so this table is what gives the training set — and the coaching — any vowel coverage at all. |

### 4.4 Feedback generation (Stage 2)

| Code | Role and importance |
| --- | --- |
| [`generator.py` → `SYSTEM_PROMPT`](app/services/feedback/generator.py#L24-L175) | **The behavioural contract** for the feedback model: per-list rules ("name *every* missing word"), the two substitution phrasings, verbatim-hint requirements, grouping of shared hints, and nine worked examples. This prompt is the specification the fine-tuning data was generated against. |
| [`generator.py` → `_report_payload()` / `build_model_messages()`](app/services/feedback/generator.py#L186-L235) | **Single source of truth for the prompt format**, imported by both the runtime and the training scripts — so the model is always served exactly the shape it was trained on. |
| [`generator.py` → `_missing_required_coverage()` / `_ensure_coverage()`](app/services/feedback/generator.py#L253-L321) | **Post-generation verifier.** Checks that every concrete mistake and every required hint appears in the generated text; appends template sentences or rejects the reply otherwise. This is what makes an LLM safe to put in a graded learning loop. |
| [`generator.py` → `generate_feedback()`](app/services/feedback/generator.py#L324-L362) | Calls the model service and **degrades gracefully**: missing config, timeout, malformed response or empty text all fall back to the deterministic renderer. The `/asr` endpoint cannot fail because the model is down. |
| [`templates.py` → `render_point()`](app/services/feedback/templates.py#L16-L60) | **Deterministic renderer** mapping each feedback-point kind to a spoken sentence, mirroring the model's phrasing so the fallback is indistinguishable in structure. |
| [`schemas/asr.py` → `ErrorReport`](app/schemas/asr.py#L156-L177) | **The contract between the two stages.** Everything Stage 1 decided, in one JSON-serializable object — the reason the model can be swapped, retrained or removed without touching any grading logic. |

### 4.5 API, persistence and progress

| Code | Role and importance |
| --- | --- |
| [`routes/asr.py` → `POST /asr`](app/api/routes/asr.py#L25-L123) | **The main endpoint.** Validates the upload, resolves the target sentence **server-side from the DB** (never trusting the client with the text it is graded against), then runs normalize → Azure → Stage 1 → Stage 2 → persist. |
| [`auth.py` → `get_current_user()`](app/api/auth.py#L18-L89) | **Firebase ID-token verification** as a FastAPI dependency, mirroring the Firebase UID into the local `users` table on first request so progress foreign keys always resolve. Every non-health route depends on it. |
| [`progress_service.py` → `record_attempt()`](app/services/progress_service.py#L18-L83) | Appends the attempt to the immutable history table, then recomputes lesson status from **distinct sentences completed in the current run** — the mechanism behind resume and per-run progress. |
| [`badge_rules.py` → `BADGE_RULES`](app/services/badge_rules.py#L29-L206) | **Rule engine as a registry of pure predicates** keyed by stable `badge_code`. Rules only read; display copy can change freely without breaking earning logic. |
| [`badge_service.py` → `evaluate_and_award_badges()`](app/services/badge_service.py#L21-L77) | **Idempotent, two-pass awarding.** Ordinary badges first, then meta badges that depend on how many badges are already earned; per-badge commits so a concurrent double-award loses only the duplicate insert. |
| [`models/domain.py`](app/models/domain.py#L30-L138) | SQLAlchemy entities, including the append-only `SentenceAttemptHistory` that every analytics query, streak rule and lesson score is derived from. |

### 4.6 Training and evaluation pipeline

| Code | Role and importance |
| --- | --- |
| [`generate_dataset.py` → `synthesize()`](training/generate_dataset.py#L222-L580) | **Synthetic `ErrorReport` generation.** Builds fake `AsrResult`s (missing / extra / minimal-pair substitution / phoneme + vowel mispronunciation / spelling-rule cases) and runs them through the **real Stage 1 analyzer**, guaranteeing the training inputs are byte-identical in shape to production inputs. |
| [`generate_dataset.py` → `bootstrap_feedback()`](training/generate_dataset.py#L583-L613) | **Distillation.** A teacher model (GPT-4o-mini) writes the ideal spoken feedback for each report, with layered rate-limit retries, producing the `{messages: [system, user, assistant]}` JSONL for supervised fine-tuning. |
| [`curate_dataset.py` → `_is_bad()`](scripts/curate_dataset.py#L94-L168) | **Automatic dataset curation.** Drops samples with markdown, excessive length, repetition, invented quoted words, or negative wording on a perfect attempt. Dataset quality control is what makes a 3.8 B model behave. |
| [`train.py` → LoRA config and merge](training/train.py#L70-L136) | **QLoRA fine-tune** of Phi-3.5-mini (4-bit NF4 base, `r=32`, `alpha=64`) targeting Phi-3's *fused* module names so the adapter survives GGUF conversion, followed by an fp16 merge for serving. Both details are the kind that silently produce a broken model if wrong. |
| [`evaluate.py` → `faithfulness()`](training/evaluate.py#L46-L83) | **Task-specific evaluation.** Because the input is structured, correctness is checkable: required-item coverage, no invented quoted words, no negative wording on a perfect attempt — plus optional BERTScore against references. Reuses the production matcher so "covered" means the same in both places. |

### 4.7 Non-functional test harnesses

| Code | Role and importance |
| --- | --- |
| [`measure_recognition_accuracy.py`](scripts/measure_recognition_accuracy.py#L1-L98) | Measures recognition accuracy against labelled recordings (a `WRONG …` filename prefix marks a deliberate mistake): overall accuracy ≥ 85 %, mistake-detection rate ≥ 80 %, false positives ≤ 10 %. |
| [`measure_response_times.py`](scripts/measure_response_times.py#L1-L107) | Times every user-visible action across full lesson runs against the 3-second requirement (≥ 80 % of actions under threshold) and emits a CSV plus a chart. |
| [`run_book_test_scenarios.py`](scripts/run_book_test_scenarios.py#L1-L118) | Automates the API-observable acceptance scenarios (lesson selection, resume, achievements, authenticated isolation) and explicitly lists the device-only scenarios it does *not* cover, so nothing is reported as passing untested. |

---

## 5. Repository layout

```text
app/
  api/
    auth.py            Firebase token verification dependency
    routes/            categories · lessons · progress · asr
  core/
    config.py          all environment access, in one place
    database.py        SQLAlchemy engine / session / get_db
  models/domain.py     SQLAlchemy entities
  schemas/             Pydantic request/response + internal contracts
  services/
    asr_service.py     Azure Pronunciation Assessment
    audio_utils.py     FFmpeg normalization + locale mapping
    feedback/          Stage 1 analyzer, hint tables, Stage 2 generator
    badge_rules.py     badge predicates
    badge_service.py   idempotent awarding
    category_service.py · lesson_service.py · progress_service.py
alembic/               database migrations
scripts/               seeding, dataset curation, non-functional measurements
tests/                 pytest unit tests (analyzer, scoring, alignment, generator)
training/              dataset generation, QLoRA fine-tune, evaluation
models/                Ollama Modelfile + exported GGUF adapter
```

---

## 6. Getting started

The project is intended to run **locally**: the backend on the development machine, the Android app
on an emulator or a device on the same network.

### Prerequisites

* Python 3.11+
* A PostgreSQL database (local or hosted)
* Azure Speech resource (key + region)
* Firebase project with a service-account JSON
* Optional: [Ollama](https://ollama.com) or a vLLM host for the fine-tuned feedback model

### Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

### Configure

Create `.env` in the repository root:

```ini
DATABASE_URL=postgresql+psycopg2://user:password@host/dbname

AZURE_SPEECH_KEY=your-azure-speech-key
AZURE_SPEECH_REGION=westeurope

# Stage 2 model — OpenAI-compatible endpoint. Omit to use the template fallback.
FEEDBACK_MODEL_URL=http://localhost:11434/v1
FEEDBACK_MODEL_NAME=aiavatar-feedback
FEEDBACK_MODEL_TIMEOUT=8.0

# Only needed to regenerate the training dataset
OPENAI_API_KEY=sk-...
```

Place the Firebase service-account file at `firebase-credentials.json` in the repository root.
Both files contain secrets and must stay untracked.

### Database

```bash
alembic upgrade head                        # create schema
python -m scripts.populate_initial_data     # seed categories, lessons, sentences, badges
```

### Run

```bash
uvicorn app.main:app --reload
```

Interactive API docs: <http://localhost:8000/docs> · health check: `GET /`.
The Android emulator reaches the host machine at `http://10.0.2.2:8000/`.

### Serve the feedback model locally (optional)

```bash
ollama create aiavatar-feedback -f models/Modelfile
ollama serve
```

Then point `FEEDBACK_MODEL_URL` at Ollama's OpenAI-compatible endpoint. If the model is
unavailable the backend automatically falls back to deterministic templates — the app keeps
working, feedback just becomes less varied.

---

## 7. API reference

All routes except `/` require `Authorization: Bearer <firebase-id-token>`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Health check |
| `GET` | `/categories/` | Categories with per-category progress |
| `GET` | `/categories/{category_id}` | Category details + lessons + progress |
| `POST` | `/lessons/{lesson_id}/start` | Start (or resume) a run; returns `run_id` |
| `GET` | `/lessons/{lesson_id}` | Lesson metadata + sentences completed in the current run |
| `GET` | `/lessons/{lesson_id}/sentences` | Ordered sentences for practice |
| `POST` | `/lessons/{lesson_id}/complete` | Finalize run, compute score, award badges |
| `POST` | `/asr` | **Pronunciation assessment** — multipart `file` + `sentence_id` + `run_id` |
| `GET` | `/progress/overview` | Average score, completed lessons, badges, recent achievements |
| `GET` | `/progress/categories` | Per-category achievement breakdown |
| `GET` | `/progress/badges` | Full badge catalog with achieved state |
| `GET` | `/progress/badges/unseen` | Badges earned but not yet celebrated in the UI |
| `POST` | `/progress/badges/seen` | Acknowledge celebration shown |

---

## 8. Data model

```mermaid
erDiagram
    User ||--o{ UserLessonProgress : has
    User ||--o{ SentenceAttemptHistory : logs
    User ||--o{ UserBadge : earns
    Category ||--o{ Lesson : contains
    Lesson ||--o{ Sentence : contains
    Lesson ||--o{ UserLessonProgress : tracks
    Sentence ||--o{ SentenceAttemptHistory : practiced_in
    Badge ||--o{ UserBadge : awarded_as
```

Two identifiers are deliberately **stable machine codes**, decoupled from display text:
`Category.category_code` and `Badge.badge_code`. Titles and descriptions can be rewritten or
localized without breaking any badge rule — see [`domain.py`](app/models/domain.py#L39-L91).

`SentenceAttemptHistory` is **append-only**: every attempt is kept, tagged with the `run_id` of the
lesson session it belongs to. Lesson scores, category averages, streaks and progress are all
derived from it rather than stored as mutable counters.

---

## 9. Testing

### Unit tests

```bash
pytest -q          # 45 tests, no network or database required
```

| File | Covers |
| --- | --- |
| [`tests/test_alignment.py`](tests/test_alignment.py) | Edit-distance alignment: matches, substitutions, omissions, insertions, tokenization |
| [`tests/test_scoring.py`](tests/test_scoring.py) | The score formula, penalty behaviour, clamping, pass threshold |
| [`tests/test_analysis.py`](tests/test_analysis.py) | Stage 1 end to end: detection thresholds, NBest confidence gates, prioritization, capping, polish tips |
| [`tests/test_generator.py`](tests/test_generator.py) | Prompt construction, coverage verification, fallback behaviour |

Fixtures in [`tests/factories.py`](tests/factories.py) build synthetic `AsrResult`s, so the analyzer
is tested without ever calling Azure.

### Non-functional measurements

These scripts require a running backend and real credentials — see
[section 4.7](#47-non-functional-test-harnesses) for what each one asserts:

```bash
python scripts/measure_recognition_accuracy.py --email … --password … --dir "…/recordings"
python scripts/measure_response_times.py       --email … --password … --audio sample.wav --runs 5
python scripts/run_book_test_scenarios.py      --email … --password …
```

---

## 10. Model training pipeline

```mermaid
flowchart LR
    S["Practice sentences<br/>DB + phonetically-rich file"] --> SYN["synthesize()<br/>fake AsrResults"]
    SYN --> A1["Real Stage 1 analyzer"]
    A1 --> RPT["ErrorReport JSON"]
    RPT --> TEACH["Teacher model<br/>ideal feedback"]
    TEACH --> CUR["curate_dataset.py<br/>automatic filtering"]
    CUR --> JSONL["train.jsonl / val.jsonl"]
    JSONL --> FT["QLoRA fine-tune<br/>Phi-3.5-mini"]
    FT --> MERGE["Merge adapter → fp16 / GGUF"]
    MERGE --> SERVE["Ollama / vLLM endpoint"]
    SERVE --> EVAL["evaluate.py<br/>faithfulness + BERTScore"]
```

Current dataset: **4,734 training** and **824 validation** examples (`training/data/*.jsonl`).
Training dependencies are separate from the app's — install `training/requirements.txt` on the GPU
machine only.

```bash
python training/generate_dataset.py --out training/data --limit 0
python scripts/curate_dataset.py
python training/train.py --data training/data --out training/out
python training/evaluate.py --endpoint http://localhost:8000/v1 --bertscore
```

[`training/train_in_colab.ipynb`](training/train_in_colab.ipynb) runs the same fine-tune on a Colab
GPU for those without local hardware.

---

## 11. Known limitations

* **Time-based badge rules operate in UTC** — streaks and the Early Bird / Night Owl windows may be
  off for learners in other timezones ([`badge_rules.py`](app/services/badge_rules.py#L1-L13)).
* **Spelling-rule error buckets stay empty at runtime.** The silent-letter, hard/soft-C and cluster
  tables exist and are used by the training generator, but Stage 1 only emits them given explicit
  evidence, which the current ASR output does not provide — so they are reported as ordinary
  mispronunciations ([`analysis.py`](app/services/feedback/analysis.py#L310-L326)).
* **`daily_streak` in `/progress/overview` returns a placeholder `0`**; the badge engine computes
  real streaks separately from attempt history.
* **CORS is fully open and ASR debug printing is enabled** — appropriate for this local development
  setup, not for public hosting.
