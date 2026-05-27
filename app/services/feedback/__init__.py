"""Two-stage feedback engine.

Stage 1 (deterministic, our code): alignment + analysis + scoring turn Azure's
raw assessment into a structured ErrorReport and a final score.
Stage 2: a fine-tuned model (or a template fallback) turns that report into
spoken feedback.
"""
