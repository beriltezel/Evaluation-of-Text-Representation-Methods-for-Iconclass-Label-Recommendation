# Evaluation-of-Text-Representation-Methods-for-Iconclass-Label-Recommendation
Bachelor thesis repository of Beril Tezel

This repository contains the final versions of the search methods and of the evaluation pipeline for comparing different text representation methods for Iconclass label recommendation. Older versions of the code are kept separately in the `older_versions` folder, so the main files in the repository are easier to follow.

## Input files

`ground_truth.csv`: the 52 queries and the Iconclass notations that are relevant for each of them. The file was converted from the Excel ground-truth file, uses `;` as the separator, and its first row is a header. The first column contains the query text, the following columns contain the relevant notations.

`used_notation_keys.txt`: the notation and key combinations that users have searched for in Iconclass, used to filter the search collection.

## Final files

`build_iconclass_hierarchy_final.py`: this file builds the Iconclass hierarchy database used in the SBERT and the evaluation steps. The database contains two SQL tables. The table `iconclass` holds the columns `notation`, `lang`, `label`, `parent`, and `depth`, where `notation` is the Iconclass code, `lang` is the language of the label, `label` is the textual Iconclass label, `parent` is the parent notation in the hierarchy, and `depth` is the number of nodes on the path of the notation. The table `hierarchy` holds `notation`, `parent`, and `depth` for every node on those paths. The `hierarchy` table is the one read for calculating Wu-Palmer similarity later.

The script also uses `used_notation_keys.txt`. In the final setup, normal Iconclass notations are kept, but notations with additions such as `(+...)` are only kept if they are listed in `used_notation_keys.txt`. This keeps the searchable Iconclass set consistent across the models.

`bm25_final.py`: this file contains the final BM25 search version used for evaluation. It reads the notations and labels directly from the Iconclass package and applies the same filtering with `used_notation_keys.txt`, so its candidate set is identical to the one stored in the database. It loads the ground truth from the CSV file, runs BM25 for every query, and saves the ranked results.

`sbert_build_embeddings_final.py`: this file builds the SBERT embeddings from the `iconclass` table of the hierarchy database, using the English labels. The embeddings, the metadata and the model name are saved in the folder `sbert_data`, so that the search file can use them later without rebuilding them every time.

`sbert_search_final.py`: this file contains the final SBERT search version used for evaluation. It loads the saved SBERT embeddings, reads the ground-truth CSV file, runs the search for every query, and saves the ranked results.

`Gemma_final.ipynb`: this notebook contains the final EmbeddingGemma search version used for evaluation. It runs in Google Colab and reads the notations and labels from the Iconclass package with the same `used_notation_keys.txt` filtering. It stores the label embeddings in a DuckDB table, reads the ground-truth CSV file, runs the search for every query, and saves the ranked results.

`Siglip_final.ipynb`: this notebook contains the final SigLIP 2 search version used for evaluation. It runs in Google Colab and reads the notations and labels from the Iconclass package with the same `used_notation_keys.txt` filtering. It stores the label embeddings in a DuckDB table, reads the ground-truth CSV file, runs the search for every query, and saves the ranked results.

All final model files read the ground truth from `ground_truth.csv`.

All four model files append their output to a JSONL file named `model_results.jsonl`, so an existing file has to be deleted before a new complete run. BM25 and SBERT are run locally and write the file in the repository folder. That file is then uploaded to the Google Drive folder of the notebooks, where EmbeddingGemma and SigLIP 2 append their records to it. The completed file is downloaded again and replaces the local file, which at that point holds only the BM25 and SBERT records, so that the evaluation runs on the version holding all four methods. Each line contains one query result, including the run ID, the timestamp, the model name, the number of retrieved notations, the query, the ground-truth codes, the predicted codes with their ranks, labels, and scores, and the ground-truth codes that are missing from the search collection.

`evaluation_final.py`: this file reads the saved model outputs from `model_results.jsonl` and evaluates the results. It also uses the Iconclass hierarchy database for the hierarchy-based metric.

The current evaluation metrics are:

- Precision
- Recall
- F1-Score
- R-Precision
- Mean Average Precision (calculated from the average precision values of the individual queries)
- Wu-Palmer Similarity from the ground truth to the predictions
- Wu-Palmer Similarity from the predictions to the ground truth
- The mean of the two Wu-Palmer values

For Wu-Palmer, the script calculates the similarity between predicted and ground-truth Iconclass codes using their position in the hierarchy.

The evaluation results are saved in `evaluation_results.jsonl`. The script also writes the graphs of the run into a timestamped folder under `evaluation_graphs`, each once without a title and once with the title below the plot, together with a `metric_values.txt` file that lists the mean values by method and the values per query.

## Run order

1. `build_iconclass_hierarchy_final.py` builds `iconclass_hierarchy.db`.
2. `bm25_final.py`, then `sbert_build_embeddings_final.py` followed by `sbert_search_final.py`.
3. `Gemma_final.ipynb` and `Siglip_final.ipynb` in Google Colab. Both need a GPU runtime and expect `used_notation_keys.txt` and `ground_truth.csv` in `/content/drive/MyDrive/Colab Notebooks/`. EmbeddingGemma is a gated model, so a Hugging Face login is required. The completed `model_results.jsonl` has to be downloaded from Drive and put into the repository folder afterwards.
4. `evaluation_final.py` computes the metrics of all four methods.
