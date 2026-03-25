"""
Train the SVA logistic classifier.

Maps the 4 SVA dimensions -> P(AI-written).
Run offline; serialise to models/sva_classifier.pkl.
SVAEngine._classify_ai() will load this at runtime via:
    clf = joblib.load('models/sva_classifier.pkl')
    return clf.predict_proba([[sem_v, lex_b, tok_e, stylo]])[0][1]

Labelled corpus sources
-----------------------
Human-written (label=0):
  - SHRM / Indeed cover letter samples
  - Kaggle "Resume Dataset" open-source letters
  - Manually collected community samples

AI-written (label=1):
  - GPT-4o generated: "Write a cover letter for a senior ML engineer role"
  - Claude 3.5 Sonnet generated (same prompt)
  - Gemini 1.5 Pro generated (same prompt)

Recommended: >= 500 samples per class for reliable calibration.

Usage
-----
# Generate synthetic dev corpus and train
python scripts/train_sva_classifier.py --synth --data data/labelled_letters.jsonl --out models/sva_classifier.pkl

# Train on real corpus
python scripts/train_sva_classifier.py --data data/labelled_letters.jsonl --out models/sva_classifier.pkl
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss

from sva.engine import SVAEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def extract_features(texts: list[str], engine: SVAEngine) -> np.ndarray:
    """
    Extract the 4 SVA feature dimensions from raw text.

    Features
    --------
    0: sem_velocity_sigma   — sigma of |delta_embed| per sentence
    1: lexical_burstiness   — variance of type-token ratio across windows
    2: token_entropy_sigma  — sigma of per-sentence perplexity
    3: stylometric_drift    — sigma of sentence-length distribution

    AI text signature: all four values uniformly LOW.
    Human text signature: all four values measurably variable.
    """
    features: list[list[float]] = []
    for i, text in enumerate(texts):
        if i % 50 == 0:
            logger.info("Extracting features %d / %d", i, len(texts))
        sentences = engine._split_sentences(text)
        features.append([
            engine._semantic_velocity_sigma(sentences),
            engine._lexical_burstiness(sentences),
            engine._token_entropy_sigma(sentences),
            engine._stylometric_drift(sentences),
        ])
    return np.array(features)


def load_corpus(path: str) -> tuple[list[str], list[int]]:
    """Load JSONL corpus. Each line: {"text": "...", "label": 0|1}"""
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(int(obj["label"]))
    logger.info(
        "Loaded %d samples — %d human / %d AI",
        len(labels), labels.count(0), labels.count(1),
    )
    return texts, labels


def train(data_path: str, output_path: str) -> None:
    engine = SVAEngine()
    texts, labels = load_corpus(data_path)
    X = extract_features(texts, engine)
    y = np.array(labels)

    # StandardScaler is critical: sem_velocity_sigma (~0.05) and
    # token_entropy_sigma (~20.0) have very different natural scales.
    # LogisticRegression is scale-sensitive.
    #
    # CalibratedClassifierCV ensures P(AI) is a real probability,
    # not just a ranking score. HR will interpret it literally.
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",  # Handles corpus imbalance
            max_iter=1000,
            random_state=42,
        )),
    ])
    clf = CalibratedClassifierCV(base, cv=5, method="sigmoid")

    # 5-fold stratified cross-validation
    logger.info("Running 5-fold stratified CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    brier = cross_val_score(clf, X, y, cv=cv, scoring="neg_brier_score")
    logger.info("CV ROC-AUC:     %.3f ± %.3f", auc.mean(), auc.std())
    logger.info("CV Brier score: %.3f ± %.3f", -brier.mean(), brier.std())

    # Final fit on full corpus
    clf.fit(X, y)
    y_pred  = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]
    logger.info("\n%s", classification_report(y, y_pred, target_names=["human", "AI"]))
    logger.info("Train ROC-AUC: %.4f", roc_auc_score(y, y_proba))
    logger.info("Train Brier:   %.4f", brier_score_loss(y, y_proba))

    # Log feature coefficients for interpretability
    inner = clf.calibrated_classifiers_[0].estimator.named_steps["clf"]
    names = ["sem_velocity_sigma", "lexical_burstiness", "token_entropy_sigma", "stylometric_drift"]
    logger.info("\nLogReg coefficients (negative = more AI-like at high value):")
    for n, c in zip(names, inner.coef_[0]):
        logger.info("  %-25s  %+.4f", n, c)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)
    logger.info("\nSaved to: %s", output_path)


def generate_synthetic_corpus(output_path: str, n_per_class: int = 300) -> None:
    """
    Generate synthetic labelled corpus for dev/testing.
    Human letters have high variance; AI letters have low variance across
    sentence length, vocabulary distribution, and semantic transitions.
    Replace with real letters for production training.
    """
    import random, string

    def random_sentence(mean: float, std: float) -> str:
        n = max(3, int(random.gauss(mean, std)))
        return " ".join(
            "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 12)))
            for _ in range(n)
        ) + "."

    def make_letter(ai: bool) -> str:
        paras = []
        for _ in range(random.randint(3, 6)):
            n_sents = random.randint(3, 7)
            if ai:
                sents = [random_sentence(18, 2) for _ in range(n_sents)]
            else:
                sents = [random_sentence(15, 8) for _ in range(n_sents)]
            paras.append(" ".join(sents))
        return "\n\n".join(paras)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for _ in range(n_per_class):
            f.write(json.dumps({"text": make_letter(False), "label": 0}) + "\n")
        for _ in range(n_per_class):
            f.write(json.dumps({"text": make_letter(True),  "label": 1}) + "\n")
    logger.info("Wrote %d synthetic samples to %s", n_per_class * 2, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVA AI-detection classifier")
    parser.add_argument("--data",  default="data/labelled_letters.jsonl",
                        help="Path to labelled JSONL corpus")
    parser.add_argument("--out",   default="models/sva_classifier.pkl",
                        help="Output path for serialised classifier")
    parser.add_argument("--synth", action="store_true",
                        help="Generate synthetic corpus before training")
    args = parser.parse_args()

    if args.synth:
        generate_synthetic_corpus(args.data)
    train(args.data, args.out)
