# Explanation of `evaluation_results.jsonl`

The file `evaluation_results.jsonl` contains the evaluation results for the model outputs saved in `model_results.jsonl`.

Each line is one JSON object, meaning one evaluated query for one model. For example, the query `bridge` will have one line for BM25, one line for SBERT, one line for Gemma, and one line for SigLIP. This makes it possible to compare how each model performed on the same query.

## Main fields

`evaluation_timestamp`: the time when the evaluation record was created.

`source_run_id`: the run ID from the original model output in `model_results.jsonl`. This helps connect the evaluation result back to the model run.

`model`: the model that was evaluated, one of `bm25`, `sbert`, `gemma`, or `siglip`.

`model_name`: the exact name of the model that produced the results. The value is empty for BM25, since BM25 is a ranking function and not a named model.

`query`: the query text from the ground-truth CSV file.

`top_n`: the number of retrieved results that were evaluated.

`ground_truth_codes`: the relevant Iconclass notations from the ground-truth CSV file.

`predicted_codes`: the Iconclass notations returned by the model, in ranked order.

## Evaluation metrics

`precision`: the fraction of predicted codes that are also in the ground truth.

`recall`: the fraction of ground-truth codes that were retrieved by the model.

`f1`: the harmonic mean of precision and recall.

`r_precision`: precision calculated at R, where R is the number of ground-truth codes for that query.

`ap`: the average precision of this single query. It measures whether the correct results appear high in the ranked prediction list. The mean average precision is the mean of these values over all queries and is not part of this file.

`wu_palmer_gt_to_pred`: for each ground-truth code, the script finds the closest predicted code using Wu-Palmer similarity and averages the scores. This shows how well the predictions cover the ground truth hierarchically.

`wu_palmer_pred_to_gt`: for each predicted code, the script finds the closest ground-truth code using Wu-Palmer similarity and averages the scores. This shows how close the predictions are to the ground truth hierarchically.

`wu_palmer_mean`: the average of `wu_palmer_gt_to_pred` and `wu_palmer_pred_to_gt`.

## Other fields

`ground_truth_codes_missing_from_corpus`: ground-truth codes that were not present in the filtered Iconclass candidate set used by the model. This is included as a check for possible ground-truth/corpus mismatches, such as notation typos, formatting differences, or codes that are outside the final candidate set. Ideally, this list should be empty.

## Notes

The evaluation results are created from `model_results.jsonl`, while the Wu-Palmer scores also use the Iconclass hierarchy database.

The mean values of all metrics by method are written to `metric_values.txt` in the timestamped folder under `evaluation_graphs`, together with the graphs of the run.