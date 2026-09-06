import os
import json
import sqlite3
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt

#This evaluation needs these files to be present in the same folder in order to function properly:
# -> iconclass_hierarchy.db
# -> ground_truth.csv
# -> model_results.jsonl


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_RESULTS_JSONL_PATH = os.path.join(BASE_DIR, "model_results.jsonl")
HIERARCHY_DB_PATH = os.path.join(BASE_DIR, "iconclass_hierarchy.db")
EVALUATION_RESULTS_JSONL_PATH = os.path.join(BASE_DIR, "evaluation_results.jsonl")

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
GRAPH_DIR = os.path.join(BASE_DIR, "evaluation_graphs", RUN_TIMESTAMP)

NICE_MODEL = {"bm25": "BM25", "sbert": "SBERT",
              "gemma": "EmbeddingGemma", "siglip": "SigLIP 2"}

NICE_METRIC = {"precision": "Precision", "recall": "Recall",
               "f1": "$F_1$-Score", "r_precision": "R-Precision",
               "ap": "MAP", "wu_palmer_mean": "Wu-Palmer mean",
               "wu_palmer_G_to_P": r"Wu-Palmer $G \rightarrow P$",
               "wu_palmer_P_to_G": r"Wu-Palmer $P \rightarrow G$",
               "wu_palmer_gt_to_pred": r"Wu-Palmer $G \rightarrow P$",
               "wu_palmer_pred_to_gt": r"Wu-Palmer $P \rightarrow G$"}

NICE_METRIC_TITLE = {"precision": "precision", "recall": "recall",
                     "f1": "$F_1$-score", "r_precision": "R-precision",
                     "ap": "average precision",
                     "wu_palmer_mean": "Wu-Palmer score",
                     "wu_palmer_G_to_P": r"Wu-Palmer $G \rightarrow P$",
                     "wu_palmer_P_to_G": r"Wu-Palmer $P \rightarrow G$",
                     "wu_palmer_gt_to_pred": r"Wu-Palmer $G \rightarrow P$",
                     "wu_palmer_pred_to_gt": r"Wu-Palmer $P \rightarrow G$"}


def save_plot(filename, title):
    """Save the current figure twice: once without a title and once with the
    title placed below the plot."""
    base, extension = os.path.splitext(filename)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, filename), dpi=300)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.figtext(0.5, 0.015, title, ha="center", va="bottom", fontsize=12)
    plt.savefig(os.path.join(GRAPH_DIR, base + "_with_title" + extension), dpi=300)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def precision_recall_f1(predicted_codes, ground_truth_codes):
    predicted_set = set(predicted_codes)
    ground_truth_set = set(ground_truth_codes)

    if not predicted_set:
        precision = 0.0
    else:
        precision = len(predicted_set & ground_truth_set) / len(predicted_set)

    if not ground_truth_set:
        recall = 0.0
    else:
        recall = len(predicted_set & ground_truth_set) / len(ground_truth_set)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def r_precision(predicted_codes, ground_truth_codes):
    r = len(ground_truth_codes)

    if r == 0:
        return 0.0

    top_r = predicted_codes[:r]
    return len(set(top_r) & set(ground_truth_codes)) / r


def average_precision(predicted_codes, ground_truth_codes):
    ground_truth_set = set(ground_truth_codes)

    if not ground_truth_set:
        return 0.0

    hits = 0
    precision_sum = 0.0
    already_found = set()

    for rank, code in enumerate(predicted_codes, start=1):
        if code in ground_truth_set and code not in already_found:
            hits += 1
            already_found.add(code)
            precision_sum += hits / rank

    return precision_sum / len(ground_truth_set)


class IconclassHierarchy:
    def __init__(self, db_path):
        self.con = sqlite3.connect(db_path)
        self.cache = {}

    def get_parent_depth(self, code):
        if code in self.cache:
            return self.cache[code]

        cur = self.con.cursor()
        cur.execute(
            """
            SELECT parent, depth
            FROM hierarchy
            WHERE notation=?
            LIMIT 1
            """,
            (code,)
        )

        row = cur.fetchone()

        if row is None:
            self.cache[code] = None
            return None

        parent, depth = row
        self.cache[code] = (parent, depth)
        return parent, depth

    def ancestors(self, code):
        result = {}
        current = code

        while current:
            parent_depth = self.get_parent_depth(current)

            if parent_depth is None:
                break

            parent, depth = parent_depth
            result[current] = depth
            current = parent

        return result

    def wu_palmer(self, code1, code2):
        if code1 == code2:
            return 1.0

        info1 = self.get_parent_depth(code1)
        info2 = self.get_parent_depth(code2)

        if info1 is None or info2 is None:
            return 0.0

        _, depth1 = info1
        _, depth2 = info2

        ancestors1 = self.ancestors(code1)
        ancestors2 = self.ancestors(code2)

        common = set(ancestors1.keys()) & set(ancestors2.keys())

        if not common:
            return 0.0

        lca = max(common, key=lambda c: ancestors1[c])
        lca_depth = ancestors1[lca]

        return (2 * lca_depth) / (depth1 + depth2)

    def close(self):
        self.con.close()


def average_best_wu_palmer(predicted_codes, ground_truth_codes, hierarchy):
    if not predicted_codes or not ground_truth_codes:
        return {
            "wu_palmer_G_to_P": 0.0,
            "wu_palmer_P_to_G": 0.0,
            "wu_palmer_mean": 0.0
        }

    gt_to_pred_scores = []

    for gt_code in ground_truth_codes:
        best_score = max(
            hierarchy.wu_palmer(gt_code, pred_code)
            for pred_code in predicted_codes
        )
        gt_to_pred_scores.append(best_score)

    pred_to_gt_scores = []

    for pred_code in predicted_codes:
        best_score = max(
            hierarchy.wu_palmer(pred_code, gt_code)
            for gt_code in ground_truth_codes
        )
        pred_to_gt_scores.append(best_score)

    gt_to_pred_avg = sum(gt_to_pred_scores) / len(gt_to_pred_scores)
    pred_to_gt_avg = sum(pred_to_gt_scores) / len(pred_to_gt_scores)

    return {
        "wu_palmer_G_to_P": gt_to_pred_avg,
        "wu_palmer_P_to_G": pred_to_gt_avg,
        "wu_palmer_mean": (gt_to_pred_avg + pred_to_gt_avg) / 2
    }


def evaluate_record(record, hierarchy):
    predicted_codes = record.get("predicted_codes", [])
    ground_truth_codes = record.get("ground_truth_codes", [])

    precision, recall, f1 = precision_recall_f1(
        predicted_codes,
        ground_truth_codes
    )

    wu_palmer_scores = average_best_wu_palmer(
        predicted_codes,
        ground_truth_codes,
        hierarchy
    )

    evaluation = {
        "evaluation_timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_run_id": record.get("run_id"),
        "model": record.get("model"),
        "model_name": record.get("model_name"),
        "query": record.get("query"),
        "top_n": record.get("top_n"),
        "ground_truth_codes": ground_truth_codes,
        "predicted_codes": predicted_codes,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "r_precision": r_precision(predicted_codes, ground_truth_codes),
        "ap": average_precision(predicted_codes, ground_truth_codes),
        "wu_palmer_gt_to_pred": wu_palmer_scores["wu_palmer_G_to_P"],
        "wu_palmer_pred_to_gt": wu_palmer_scores["wu_palmer_P_to_G"],
        "wu_palmer_mean": wu_palmer_scores["wu_palmer_mean"],
        "ground_truth_codes_missing_from_corpus": record.get(
            "ground_truth_codes_missing_from_corpus",
            []
        )
    }

    return evaluation


def summarize_by_model(records):
    metrics = [
        "precision",
        "recall",
        "f1",
        "r_precision",
        "ap",
        "wu_palmer_mean",
        "wu_palmer_gt_to_pred",
        "wu_palmer_pred_to_gt"
    ]

    scores_by_model = defaultdict(lambda: defaultdict(list))

    for record in records:
        model = record.get("model")

        if not model:
            continue

        for metric in metrics:
            value = record.get(metric)

            if value is not None:
                scores_by_model[model][metric].append(value)

    models = sorted(scores_by_model.keys())
    summary = {}

    for model in models:
        summary[model] = {}

        for metric in metrics:
            values = scores_by_model[model][metric]

            if values:
                summary[model][metric] = sum(values) / len(values)
            else:
                summary[model][metric] = 0.0

    return summary, metrics, models


def plot_evaluation_results(records):
    os.makedirs(GRAPH_DIR, exist_ok=True)

    summary, metrics, models = summarize_by_model(records)

    if not models:
        print("No model scores found for plotting.")
        return

    for metric in metrics:
        values = [summary[model][metric] for model in models]

        plt.figure(figsize=(8, 5))
        plt.bar([NICE_MODEL.get(model, model) for model in models], values)
        plt.ylim(0, 1)
        plt.xlabel("Method")
        plt.ylabel(NICE_METRIC.get(metric, metric))

        save_plot(f"{metric}_by_method.png",
                  f"Mean {NICE_METRIC_TITLE.get(metric, metric)} by method")

    x = range(len(models))
    width = 0.1

    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        values = [summary[model][metric] for model in models]
        positions = [pos + (i - len(metrics) / 2) * width for pos in x]
        plt.bar(positions, values, width=width, label=NICE_METRIC.get(metric, metric))

    plt.xticks(list(x), [NICE_MODEL.get(model, model) for model in models])
    plt.ylim(0, 1)
    plt.xlabel("Method")
    plt.ylabel("Score")
    plt.legend()

    save_plot("all_metrics_by_method.png",
              "Mean scores of the four methods over the 52 queries")

    print("Graphs saved in:")
    print(GRAPH_DIR)

    plt.show()

def plot_query_performance(records):
    os.makedirs(GRAPH_DIR, exist_ok=True)

# This function creates graphs showing metric performance for each query, with one graph per metric.

    metrics = [
        "precision",
        "recall",
        "f1",
        "r_precision",
        "ap",
        "wu_palmer_mean",
        "wu_palmer_gt_to_pred",
        "wu_palmer_pred_to_gt"
    ]

    query_order = []
    model_order = []
    values_by_model_query = {}

    for record in records:
        model = record.get("model")
        query = record.get("query")

        if not model or not query:
            continue

        if query not in query_order:
            query_order.append(query)

        if model not in model_order:
            model_order.append(model)

        values_by_model_query[(model, query)] = record

    if not query_order or not model_order:
        print("No query scores found for plotting.")
        return

    x = range(len(query_order))

    for metric in metrics:
        plt.figure(figsize=(16, 7))

        for model in model_order:
            values = []

            for query in query_order:
                record = values_by_model_query.get((model, query))

                if record is None:
                    values.append(None)
                else:
                    values.append(record.get(metric, 0.0))

            plt.plot(
                x,
                values,
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=NICE_MODEL.get(model, model)
            )

        plt.xticks(list(x), query_order, rotation=75, ha="right")
        plt.ylim(0, 1.05)
        title_metric = NICE_METRIC_TITLE.get(metric, metric)
        plt.xlabel("Query")
        plt.ylabel(NICE_METRIC.get(metric, metric))
        plt.grid(True, alpha=0.3)
        plt.legend()

        save_plot(f"{metric}_per_query.png",
                  f"{title_metric[0].upper()}{title_metric[1:]} per query")

    print("Query performance graphs saved in:")
    print(GRAPH_DIR)

    plt.show()

def write_metric_values(records):
    os.makedirs(GRAPH_DIR, exist_ok=True)
    values_path = os.path.join(GRAPH_DIR, "metric_values.txt")

    summary, metrics, models = summarize_by_model(records)

    if not models:
        print("No model scores found for the value file.")
        return

    query_order = []
    records_by_model_query = {}

    for record in records:
        model = record.get("model")
        query = record.get("query")

        if not model or not query:
            continue

        if query not in query_order:
            query_order.append(query)

        records_by_model_query[(model, query)] = record

    metric_width = max(len(metric) for metric in metrics) + 2
    query_width = max([len(query) for query in query_order] + [len("query")]) + 2
    model_width = max([len(model) for model in models] + [len("ground truth")]) + 2

    lines = []
    lines.append("EVALUATION VALUES")
    lines.append(f"Run: {RUN_TIMESTAMP}")
    lines.append(f"Models: {', '.join(models)}")
    lines.append(f"Queries: {len(query_order)}")
    lines.append("")
    lines.append("1 AVERAGE VALUES BY MODEL")
    lines.append("")
    lines.append("metric".ljust(metric_width) + "".join(model.rjust(model_width) for model in models))
    lines.append("-" * (metric_width + model_width * len(models)))

    for metric in metrics:
        row = metric.ljust(metric_width)

        for model in models:
            row += f"{summary[model][metric]:.4f}".rjust(model_width)

        lines.append(row)

    lines.append("")
    lines.append("2 VALUES PER QUERY")

    for metric in metrics:
        lines.append("")
        lines.append(f"2.{metrics.index(metric) + 1} {metric}")
        lines.append("")
        lines.append("query".ljust(query_width) + "".join(model.rjust(model_width) for model in models))
        lines.append("-" * (query_width + model_width * len(models)))

        for query in query_order:
            row = query.ljust(query_width)

            for model in models:
                record = records_by_model_query.get((model, query))

                if record is None:
                    row += "-".rjust(model_width)
                else:
                    row += f"{record.get(metric, 0.0):.4f}".rjust(model_width)

            lines.append(row)

    lines.append("")
    lines.append("3 NUMBER OF RESULTS PER QUERY")
    lines.append("")
    lines.append(
        "query".ljust(query_width)
        + "ground truth".rjust(model_width)
        + "".join(model.rjust(model_width) for model in models)
    )
    lines.append("-" * (query_width + model_width * (len(models) + 1)))

    for query in query_order:
        row = query.ljust(query_width)
        ground_truth_size = 0

        for model in models:
            record = records_by_model_query.get((model, query))

            if record is not None:
                ground_truth_size = len(record.get("ground_truth_codes", []))
                break

        row += str(ground_truth_size).rjust(model_width)

        for model in models:
            record = records_by_model_query.get((model, query))

            if record is None:
                row += "-".rjust(model_width)
            else:
                row += str(len(record.get("predicted_codes", []))).rjust(model_width)

        lines.append(row)

    with open(values_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Metric values written to:")
    print(values_path)

def run_evaluation():
    model_records = load_jsonl(MODEL_RESULTS_JSONL_PATH)
    hierarchy = IconclassHierarchy(HIERARCHY_DB_PATH)

    evaluation_records = []

    for record in model_records:
        evaluation = evaluate_record(record, hierarchy)
        evaluation_records.append(evaluation)

    hierarchy.close()

    write_jsonl(evaluation_records, EVALUATION_RESULTS_JSONL_PATH)

    print("Finished evaluation.")
    print(f"Model result records processed: {len(model_records)}")
    print(f"Evaluation results written to: {EVALUATION_RESULTS_JSONL_PATH}")

    write_metric_values(evaluation_records)
    plot_evaluation_results(evaluation_records)
    plot_query_performance(evaluation_records)

if __name__ == "__main__":
    run_evaluation()