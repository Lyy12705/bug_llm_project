"""Train REP- related-report weights from Bugzilla duplicate links.

若 bug A 的 `dupe_of` 指向 bug B，程式會把 A/B 視為 related pair，
再用非 duplicate 歷史 bug 當 negative samples，學習 REP- 相似度特徵權重。
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from build_features import (
    bm25f_ext_matrices,
    drone_bigram_analyzer,
    drone_text_analyzer,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "eclipse_bug_reports_clean.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "rep_minus_weights.json")

DEFAULT_CONFIG = {
    "summary_weight": 2.0,
    "description_weight": 1.0,
    "bigram_summary_weight": 2.0,
    "bigram_description_weight": 1.0,
    "unigram_feature_weight": 0.9,
    "bigram_feature_weight": 0.2,
    "product_weight": 2.0,
    "component_weight": 0.0,
    "severity_weight": 0.0,
    "k1_unigram": 1.2,
    "k3_unigram": 100.0,
    "summary_b_unigram": 0.75,
    "description_b_unigram": 0.75,
    "k1_bigram": 1.2,
    "k3_bigram": 100.0,
    "summary_b_bigram": 0.75,
    "description_b_bigram": 0.75,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune REP- linear feature weights from duplicate links using RankNet-style SGD."
    )
    parser.add_argument("--input", default=INPUT_PATH, help="Cleaned CSV with dupe_of column.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output REP- weights JSON.")
    parser.add_argument("--max-features", type=int, default=3000, help="Vocabulary size for unigram and bigram fields.")
    parser.add_argument("--negative-samples", type=int, default=5, help="Irrelevant reports sampled per duplicate pair.")
    parser.add_argument("--rounds", type=int, default=2, help="SGD passes over pairwise ranking instances.")
    parser.add_argument("--learning-rate", type=float, default=0.02, help="SGD learning rate.")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 penalty for feature weights.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--summary-weight", type=float, default=DEFAULT_CONFIG["summary_weight"], help="BM25Fext summary field weight.")
    parser.add_argument("--description-weight", type=float, default=DEFAULT_CONFIG["description_weight"], help="BM25Fext description field weight.")
    parser.add_argument(
        "--bigram-summary-weight",
        type=float,
        default=DEFAULT_CONFIG["bigram_summary_weight"],
        help="BM25Fext bigram summary field weight.",
    )
    parser.add_argument(
        "--bigram-description-weight",
        type=float,
        default=DEFAULT_CONFIG["bigram_description_weight"],
        help="BM25Fext bigram description field weight.",
    )
    parser.add_argument("--k1-unigram", type=float, default=DEFAULT_CONFIG["k1_unigram"], help="BM25Fext k1 for unigram.")
    parser.add_argument("--k3-unigram", type=float, default=DEFAULT_CONFIG["k3_unigram"], help="BM25Fext k3 for unigram.")
    parser.add_argument("--summary-b-unigram", type=float, default=DEFAULT_CONFIG["summary_b_unigram"], help="BM25Fext summary b for unigram.")
    parser.add_argument(
        "--description-b-unigram",
        type=float,
        default=DEFAULT_CONFIG["description_b_unigram"],
        help="BM25Fext description b for unigram.",
    )
    parser.add_argument("--k1-bigram", type=float, default=DEFAULT_CONFIG["k1_bigram"], help="BM25Fext k1 for bigram.")
    parser.add_argument("--k3-bigram", type=float, default=DEFAULT_CONFIG["k3_bigram"], help="BM25Fext k3 for bigram.")
    parser.add_argument("--summary-b-bigram", type=float, default=DEFAULT_CONFIG["summary_b_bigram"], help="BM25Fext summary b for bigram.")
    parser.add_argument(
        "--description-b-bigram",
        type=float,
        default=DEFAULT_CONFIG["description_b_bigram"],
        help="BM25Fext description b for bigram.",
    )
    return parser


def clean_dupe_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return "" if text.lower() in {"", "none", "nan", "0"} else text


def pair_feature(
    query_idx: int,
    doc_idx: int,
    X_doc_uni,
    X_query_uni,
    X_doc_bi,
    X_query_bi,
    products: np.ndarray,
    components: np.ndarray,
    severities: np.ndarray,
) -> np.ndarray:
    # 每個 duplicate/non-duplicate pair 轉成 REP- 訓練用的五個相似度訊號。
    unigram = float((X_query_uni[query_idx] @ X_doc_uni[doc_idx].T).toarray()[0, 0])
    bigram = float((X_query_bi[query_idx] @ X_doc_bi[doc_idx].T).toarray()[0, 0])
    same_product = float(products[query_idx] == products[doc_idx])
    same_component = float(components[query_idx] == components[doc_idx])
    same_severity = float(severities[query_idx] == severities[doc_idx])
    return np.array([unigram, bigram, same_product, same_component, same_severity], dtype=float)


def sigmoid(value: float) -> float:
    value = float(np.clip(value, -50, 50))
    return 1.0 / (1.0 + np.exp(-value))


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = pd.read_csv(args.input)
    if "dupe_of" not in df.columns:
        config = DEFAULT_CONFIG | {
            "training_pairs": 0,
            "note": "Input has no dupe_of column; using default REP- initialization weights.",
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
        print("No dupe_of column found; saved default REP- weights.")
        print(f"saved weights -> {args.output}")
        return

    df["creation_time"] = pd.to_datetime(df["creation_time"], errors="coerce")
    df = df.dropna(subset=["creation_time"]).sort_values(["creation_time", "id"]).reset_index(drop=True)
    df["dupe_of_clean"] = df["dupe_of"].map(clean_dupe_id)

    texts = df["text"].fillna("").astype(str).tolist()
    summaries = df["summary"].fillna("").astype(str).tolist()
    descriptions = df["description"].fillna("").astype(str).tolist()

    unigram_vectorizer = CountVectorizer(max_features=args.max_features, analyzer=drone_text_analyzer)
    unigram_vectorizer.fit(texts)
    bigram_vectorizer = CountVectorizer(max_features=args.max_features, analyzer=drone_bigram_analyzer)
    bigram_vectorizer.fit(texts)

    X_summary_uni = unigram_vectorizer.transform(summaries)
    X_description_uni = unigram_vectorizer.transform(descriptions)
    X_doc_uni, X_query_uni = bm25f_ext_matrices(
        X_summary_uni,
        X_description_uni,
        summary_weight=args.summary_weight,
        description_weight=args.description_weight,
        summary_b=args.summary_b_unigram,
        description_b=args.description_b_unigram,
        k1=args.k1_unigram,
        k3=args.k3_unigram,
    )

    X_summary_bi = bigram_vectorizer.transform(summaries)
    X_description_bi = bigram_vectorizer.transform(descriptions)
    X_doc_bi, X_query_bi = bm25f_ext_matrices(
        X_summary_bi,
        X_description_bi,
        summary_weight=args.bigram_summary_weight,
        description_weight=args.bigram_description_weight,
        summary_b=args.summary_b_bigram,
        description_b=args.description_b_bigram,
        k1=args.k1_bigram,
        k3=args.k3_bigram,
    )

    id_to_idx = {str(bug_id): idx for idx, bug_id in enumerate(df["id"].astype(str))}
    products = df["product"].astype(str).to_numpy()
    components = df["component"].astype(str).to_numpy()
    severities = df["severity"].astype(str).str.lower().to_numpy()
    rng = np.random.default_rng(args.seed)

    instances: list[tuple[np.ndarray, np.ndarray]] = []
    for query_idx, row in df.iterrows():
        master_id = row["dupe_of_clean"]
        if not master_id or master_id not in id_to_idx:
            continue
        rel_idx = id_to_idx[master_id]
        if rel_idx >= query_idx:
            continue
        candidate_irrelevant = np.arange(query_idx)
        candidate_irrelevant = candidate_irrelevant[candidate_irrelevant != rel_idx]
        if len(candidate_irrelevant) == 0:
            continue
        sample_size = min(args.negative_samples, len(candidate_irrelevant))
        negative_indices = rng.choice(candidate_irrelevant, size=sample_size, replace=False)
        rel_feature = pair_feature(
            query_idx, rel_idx, X_doc_uni, X_query_uni, X_doc_bi, X_query_bi, products, components, severities
        )
        for irr_idx in negative_indices:
            irr_feature = pair_feature(
                query_idx, int(irr_idx), X_doc_uni, X_query_uni, X_doc_bi, X_query_bi, products, components, severities
            )
            instances.append((rel_feature, irr_feature))

    weights = np.array(
        [
            DEFAULT_CONFIG["unigram_feature_weight"],
            DEFAULT_CONFIG["bigram_feature_weight"],
            DEFAULT_CONFIG["product_weight"],
            DEFAULT_CONFIG["component_weight"],
            DEFAULT_CONFIG["severity_weight"],
        ],
        dtype=float,
    )

    if instances:
        order = np.arange(len(instances))
        for _ in range(args.rounds):
            rng.shuffle(order)
            for idx in order:
                rel_feature, irr_feature = instances[int(idx)]
                delta = irr_feature - rel_feature
                grad = sigmoid(np.dot(weights, delta)) * delta + args.l2 * weights
                weights -= args.learning_rate * grad
                weights = np.maximum(weights, 0.0)

    config = DEFAULT_CONFIG | {
        "summary_weight": args.summary_weight,
        "description_weight": args.description_weight,
        "bigram_summary_weight": args.bigram_summary_weight,
        "bigram_description_weight": args.bigram_description_weight,
        "k1_unigram": args.k1_unigram,
        "k3_unigram": args.k3_unigram,
        "summary_b_unigram": args.summary_b_unigram,
        "description_b_unigram": args.description_b_unigram,
        "k1_bigram": args.k1_bigram,
        "k3_bigram": args.k3_bigram,
        "summary_b_bigram": args.summary_b_bigram,
        "description_b_bigram": args.description_b_bigram,
        "unigram_feature_weight": float(weights[0]),
        "bigram_feature_weight": float(weights[1]),
        "product_weight": float(weights[2]),
        "component_weight": float(weights[3]),
        "severity_weight": float(weights[4]),
        "feature_names": [
            "unigram_bm25fext",
            "bigram_bm25fext",
            "same_product",
            "same_component",
            "same_severity",
        ],
        "training_pairs": len(instances),
        "negative_samples": args.negative_samples,
        "rounds": args.rounds,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "note": (
            "Feature weights were tuned with RankNet-style SGD. BM25Fext internal parameters "
            "are exposed in this JSON but kept at defaults unless edited."
        ),
    }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    print(f"training instances: {len(instances)}")
    print(
        "weights: "
        f"unigram={weights[0]:.4f}, bigram={weights[1]:.4f}, "
        f"product={weights[2]:.4f}, component={weights[3]:.4f}, "
        f"severity={weights[4]:.4f}"
    )
    print(f"saved weights -> {args.output}")


if __name__ == "__main__":
    main()
