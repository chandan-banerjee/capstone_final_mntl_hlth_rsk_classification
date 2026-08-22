"""
Synthetic Mental Health Risk Data Generator
============================================
Generates synthetic data that mirrors the statistical distributions of
mental_health_risk_dataset (1).csv using only Python standard library.

Usage:
    python generate_synthetic_mental_health.py [--rows 25000] [--output synthetic_mental_health.csv] [--seed 42]
"""

import csv
import random
import math
import argparse

# ---------------------------------------------------------------------------
# Distributions derived from the original 25 000-row dataset
# ---------------------------------------------------------------------------

CATEGORICAL = {
    "gender":                          ["Male", "Female", "Other"],
    "gender_weights":                  [8438, 8300, 8262],

    "marital_status":                  ["Single", "Divorced", "Married"],
    "marital_status_weights":          [8161, 8457, 8382],

    "education_level":                 ["High School", "Bachelor", "Master", "PhD"],
    "education_level_weights":         [6356, 6261, 6195, 6188],

    "employment_status":               ["Employed", "Self-Employed", "Unemployed", "Student"],
    "employment_status_weights":       [6293, 6245, 6235, 6227],

    "mental_health_risk":              ["0", "1", "2"],
    "mental_health_risk_weights":      [9357, 11823, 3820],
}

# Binary columns: weight for value "1"
BINARY = {
    "panic_attack_history":            12446 / 25000,
    "family_history_mental_illness":   12607 / 25000,
    "previous_mental_health_diagnosis":12604 / 25000,
    "therapy_history":                 12541 / 25000,
    "substance_use":                   12478 / 25000,
}

# Continuous columns: (min, max, mean, stdev) — clipped to [min, max]
CONTINUOUS = {
    "age":                             (18, 60,   39.07, 12.40),
    "sleep_hours":                     (3,  10,    6.51,  2.02),
    "physical_activity_hours_per_week":(0,  15,    7.52,  4.32),
    "screen_time_hours_per_day":       (1,  12,    6.47,  3.17),
    "working_hours_per_week":          (20, 70,   45.05, 14.68),
}

# Integer score columns: (min, max, mean, stdev)
SCORES = {
    "social_support_score":            (1, 10, 5.51, 2.89),
    "work_stress_level":               (1, 10, 5.52, 2.86),
    "academic_pressure_level":         (1, 10, 5.45, 2.86),
    "job_satisfaction_score":          (1, 10, 5.47, 2.87),
    "financial_stress_level":          (1, 10, 5.50, 2.88),
    "anxiety_score":                   (1, 10, 5.51, 2.87),
    "depression_score":                (1, 10, 5.53, 2.87),
    "stress_level":                    (1, 10, 5.52, 2.87),
    "mood_swings_frequency":           (1, 10, 5.51, 2.87),
    "concentration_difficulty_level":  (1, 10, 5.49, 2.87),
}

# Column order (matches original CSV header)
COLUMNS = [
    "age", "gender", "marital_status", "education_level", "employment_status",
    "sleep_hours", "physical_activity_hours_per_week", "screen_time_hours_per_day",
    "social_support_score", "work_stress_level", "academic_pressure_level",
    "job_satisfaction_score", "financial_stress_level", "working_hours_per_week",
    "anxiety_score", "depression_score", "stress_level", "mood_swings_frequency",
    "concentration_difficulty_level", "panic_attack_history",
    "family_history_mental_illness", "previous_mental_health_diagnosis",
    "therapy_history", "substance_use", "mental_health_risk",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def gauss_continuous(rng, lo, hi, mean, stdev, decimals=1):
    """Sample from a Gaussian clipped to [lo, hi]."""
    value = rng.gauss(mean, stdev)
    return round(clamp(value, lo, hi), decimals)


def gauss_int(rng, lo, hi, mean, stdev):
    """Sample an integer from a Gaussian clipped to [lo, hi]."""
    value = rng.gauss(mean, stdev)
    return int(clamp(round(value), lo, hi))


def weighted_choice(rng, choices, weights):
    total = sum(weights)
    r = rng.uniform(0, total)
    cumulative = 0
    for choice, weight in zip(choices, weights):
        cumulative += weight
        if r <= cumulative:
            return choice
    return choices[-1]


# ---------------------------------------------------------------------------
# Row generator
# ---------------------------------------------------------------------------

def generate_row(rng):
    row = {}

    # Categorical
    for col in ["gender", "marital_status", "education_level", "employment_status", "mental_health_risk"]:
        row[col] = weighted_choice(rng, CATEGORICAL[col], CATEGORICAL[f"{col}_weights"])

    # Binary
    for col, prob in BINARY.items():
        row[col] = 1 if rng.random() < prob else 0

    # Continuous
    for col, (lo, hi, mean, stdev) in CONTINUOUS.items():
        if col == "age":
            row[col] = gauss_int(rng, lo, hi, mean, stdev)
        elif col == "working_hours_per_week":
            row[col] = gauss_int(rng, lo, hi, mean, stdev)
        else:
            row[col] = gauss_continuous(rng, lo, hi, mean, stdev, decimals=1)

    # Integer scores
    for col, (lo, hi, mean, stdev) in SCORES.items():
        row[col] = gauss_int(rng, lo, hi, mean, stdev)

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic mental health risk data")
    parser.add_argument("--rows",   type=int,   default=25000,                         help="Number of rows to generate (default: 25000)")
    parser.add_argument("--output", type=str,   default="synthetic_mental_health.csv", help="Output CSV file name")
    parser.add_argument("--seed",   type=int,   default=42,                            help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Generating {args.rows:,} synthetic rows -> {args.output}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i in range(args.rows):
            writer.writerow(generate_row(rng))

    print("Done.")


if __name__ == "__main__":
    main()


'''
How it works:

Categorical columns (gender, marital_status, education_level, employment_status, mental_health_risk) — weighted random draws from the exact original value counts
Binary flags (panic_attack_history, family_history_mental_illness, etc.) — Bernoulli draws matching the original proportion of 1s
Continuous columns (age, sleep_hours, working_hours_per_week, etc.) — Gaussian sampling (original mean/stdev) clipped to the original min/max
Integer score columns (all 1–10 scales) — Gaussian rounded to int, clipped to [1, 10]
'''