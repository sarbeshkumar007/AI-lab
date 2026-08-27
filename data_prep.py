"""
Shared data-cleaning and feature-engineering logic.

Imported by both train_model.py (training time) and app.py (inference time)
so the two never drift out of sync — whatever transformation the model was
trained on is exactly what runs on a live user input.
"""

import pandas as pd

TARGET_COL = "CARDIO_DISEASE"

# Physiologically-implausible-value bounds, used only to drop bad rows at
# training time. Not re-applied at inference time (a single user input is
# clipped, not dropped — see clip_single_input below).
BOUNDS = {
    "HEIGHT": (120, 210),
    "WEIGHT": (30, 180),
    "AP_HIGH": (80, 200),
    "AP_LOW": (50, 130),
}


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are impossible, not just unusual, for a live human being."""
    d = df.copy()
    d = d[d["AP_LOW"] <= d["AP_HIGH"]]
    for col, (lo, hi) in BOUNDS.items():
        d = d[(d[col] >= lo) & (d[col] <= hi)]
    d = d.drop_duplicates()
    return d


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add BMI, pulse pressure, and mean arterial pressure (MAP).

    These are standard clinical derived values computed from columns already
    in the raw data — they don't require anything the user/dataset doesn't
    already provide, and they encode domain knowledge the raw columns leave
    the model to rediscover on its own (e.g. that height and weight only
    matter together, not independently).
    """
    d = df.copy()
    d["BMI"] = d["WEIGHT"] / ((d["HEIGHT"] / 100) ** 2)
    d["PULSE_PRESSURE"] = d["AP_HIGH"] - d["AP_LOW"]
    d["MAP"] = d["AP_LOW"] + d["PULSE_PRESSURE"] / 3
    return d


def prepare_training_frame(df: pd.DataFrame):
    """Full pipeline: clean -> engineer -> split into X, y."""
    d = clean_training_data(df)
    d = add_engineered_features(d)
    X = d.drop(columns=[TARGET_COL])
    y = d[TARGET_COL]
    return X, y


def clip_single_input(raw: dict) -> dict:
    """Clip one live input into the same plausible range used to filter
    training data, instead of dropping it (there's no 'drop' option for a
    real prediction request). Keeps the app from getting a garbage
    prediction on a fat-fingered value.
    """
    clipped = dict(raw)
    for col, (lo, hi) in BOUNDS.items():
        clipped[col] = min(max(clipped[col], lo), hi)
    if clipped["AP_LOW"] > clipped["AP_HIGH"]:
        clipped["AP_LOW"], clipped["AP_HIGH"] = clipped["AP_HIGH"], clipped["AP_LOW"]
    return clipped
