from iconclass import init
import math
import csv
import json
import os
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

USED_NOTATION_KEYS_PATH = os.path.join(BASE_DIR, "used_notation_keys.txt")
GROUND_TRUTH_CSV_PATH = os.path.join(BASE_DIR, "ground_truth.csv")
OUTPUT_JSONL_PATH = os.path.join(BASE_DIR, "model_results.jsonl")

LANG = "en"
MODEL_NAME = "bm25"

k1 = 1.5
b = 0.75
TOP_N = 20


def load_used_notation_keys(path=USED_NOTATION_KEYS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def keep_notation(notation, used_notation_keys):
    if notation.find("(+") > 1 and notation not in used_notation_keys:
        return False
    return True


def extract_variant_notations(ic):
    variant_notations = []

    for notation, obj in ic.source._D.items():
        if notation is None:
            continue

        key = obj.get("k")
        if not key:
            continue

        for suffix in key.get("s", []):
            variant_notations.append(f"{notation}(+{suffix})")

    return variant_notations


def load_ground_truth(path=GROUND_TRUTH_CSV_PATH, has_header=True):
    ground_truth = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")

        if has_header:
            next(reader, None)

        for row in reader:
            if not row:
                continue

            query = row[0].strip()

            if not query:
                continue

            relevant_codes = [
                cell.strip()
                for cell in row[1:]
                if cell and cell.strip()
            ]

            ground_truth[query] = relevant_codes

    return ground_truth


def build_bm25_corpus(used_notation_keys_path=USED_NOTATION_KEYS_PATH):
    ic = init()
    used_notation_keys = load_used_notation_keys(used_notation_keys_path)

    corpus = []

    base_notations = [x for x in ic.source._D.keys() if x is not None]
    variant_notations = extract_variant_notations(ic)

    all_notations = sorted(set(base_notations + variant_notations))

    skipped_filtered = 0
    skipped_empty = 0
    inserted_variants = 0

    for notation in all_notations:
        if not keep_notation(notation, used_notation_keys):
            skipped_filtered += 1
            continue

        try:
            node = ic[notation]
            label = node(LANG) or ""
        except Exception:
            label = ""

        label = label.strip()

        if not label:
            skipped_empty += 1
            continue

        tokens = label.lower().split()

        corpus.append({
            "notation": notation,
            "label": label,
            "tokens": tokens
        })

        if "(+" in notation:
            inserted_variants += 1

    df = defaultdict(int)

    for item in corpus:
        for token in set(item["tokens"]):
            df[token] += 1

    avgdl = sum(len(item["tokens"]) for item in corpus) / len(corpus)
    valid_codes = set(item["notation"] for item in corpus)

    print(f"Total source notations: {len(all_notations)}")
    print(f"Filtered BM25 corpus size: {len(corpus)}")
    print(f"Inserted variants: {inserted_variants}")
    print(f"Skipped by filtering rule: {skipped_filtered}")
    print(f"Skipped empty labels: {skipped_empty}")

    return corpus, df, avgdl, valid_codes


corpus, df, avgdl, valid_codes = build_bm25_corpus()
N = len(corpus)


def bm25_score(query_tokens, doc_tokens):
    score = 0.0
    doc_len = len(doc_tokens)

    for q in query_tokens:
        f = doc_tokens.count(q)

        if f == 0:
            continue

        df_q = df.get(q, 0)

        if df_q == 0:
            continue

        idf = math.log((N - df_q + 0.5) / (df_q + 0.5) + 1)
        denom = f + k1 * (1 - b + b * (doc_len / avgdl))
        score += idf * ((f * (k1 + 1)) / denom)

    return score


def search_iconclass_bm25(query, top_n=TOP_N):
    query_tokens = query.lower().split()
    results = []

    for item in corpus:
        score = bm25_score(query_tokens, item["tokens"])

        if score > 0:
            results.append({
                "code": item["notation"],
                "label": item["label"],
                "score": float(score)
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_n]

    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    return results


def append_jsonl(record, path=OUTPUT_JSONL_PATH):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def check_ground_truth_against_corpus(ground_truth):
    missing = {}

    for query, gt_codes in ground_truth.items():
        missing_codes = [code for code in gt_codes if code not in valid_codes]

        if missing_codes:
            missing[query] = missing_codes

    return missing


def run_bm25_on_ground_truth(
    ground_truth_csv_path=GROUND_TRUTH_CSV_PATH,
    output_jsonl_path=OUTPUT_JSONL_PATH,
    top_n=TOP_N,
    has_header=True
):
    ground_truth = load_ground_truth(
        path=ground_truth_csv_path,
        has_header=has_header
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    missing = check_ground_truth_against_corpus(ground_truth)

    if missing:
        print("\nWarning: some ground-truth codes are not in the BM25 corpus.")
        print("These codes cannot be retrieved by BM25 under the shared filtering rule.")
        print(f"Queries affected: {len(missing)}\n")

    for query, gt_codes in tqdm(
        ground_truth.items(),
        total=len(ground_truth),
        desc="Running BM25",
        unit="query"
    ):
        results = search_iconclass_bm25(query, top_n=top_n)

        record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL_NAME,
            "top_n": top_n,
            "query": query,
            "ground_truth_codes": gt_codes,
            "predicted_results": results,
            "predicted_codes": [item["code"] for item in results],
            "ground_truth_codes_missing_from_corpus": missing.get(query, [])
        }

        append_jsonl(record, path=output_jsonl_path)

    print("Finished BM25 run.")
    print(f"Queries processed: {len(ground_truth)}")
    print(f"Results appended to: {output_jsonl_path}")


if __name__ == "__main__":
    run_bm25_on_ground_truth(
        ground_truth_csv_path=GROUND_TRUTH_CSV_PATH,
        output_jsonl_path=OUTPUT_JSONL_PATH,
        top_n=TOP_N,
        has_header=True
    )