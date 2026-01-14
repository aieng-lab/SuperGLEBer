import itertools
import json
import os
from collections import Counter, defaultdict, OrderedDict
from functools import partial
from typing import Set, Dict, List, Tuple, Iterable, Optional, Sequence, Any

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm, pearsonr
from evaluate import load
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import f1_score

from src.evaluation.bootstrap_aggregation import aggregate

task2metric = {
    'offensive_lang': 'macro_f1',
    'toxic_comments': 'macro_f1',

    'polarity': 'micro_f1',
    'db_aspect': 'micro_f1',
    'hotel_aspect': 'micro_f1',

    'query_ad': 'accuracy',
    'quest_ans': 'accuracy',
    'pawsx': 'accuracy',

    'webcage': 'micro_f1',
    'verbal_idioms': 'micro_f1',

    'factclaiming_comments': 'macro_f1',
    'engaging_comments': 'macro_f1',
    'argument_mining': 'macro_f1',
    'topic_relevance': 'micro_f1',
    'massive_intents': 'micro_f1',
    'nli': 'accuracy',
    'news_class': 'accuracy',

    'ner_biofid': 'micro_f1',
    'ner_europarl': 'micro_f1',
    'ner_legal': 'micro_f1',
    'ner_news': 'micro_f1',
    'ner_wiki_news': 'micro_f1',

    'up_dep': 'micro_f1',
    'up_pos': 'micro_f1',
    'massive_seq': 'micro_f1',
    'germeval_opinions': 'micro_f1',


    'similarity_pawsx': 'pearsonr',

    'mlqa': 'mean_token_f1',
    'germanquad': 'mean_token_f1',



}

all_supergleber_tasks = list(task2metric.keys())

def mean_token_f1_score(predictions, references):
    common = Counter(predictions) & Counter(references)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(predictions)
    recall = 1.0 * num_same / len(references)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def pearson_score(predictions, references):
    pearson_corr, _ = pearsonr(references, predictions)
    return pearson_corr

metrics = {
    'mean_token_f1': mean_token_f1_score,
    'accuracy': load('accuracy').compute,
    'macro_f1': partial(f1_score, average='macro'),
    'micro_f1': partial(f1_score, average='micro'),
    'pearsonr': pearson_score,
}


def bootstrap_supergleber_scores(df: pd.DataFrame, n_samples: int=1000, seed: int=42) -> list:
    """
    Compute bootstrapped SuperGLEBer task scores (and their mean) across multiple seeds.

    The function assumes `df` contains per-example predictions and gold labels for one
    or more tasks, possibly with multiple runs (seeds). It repeatedly resamples the
    examples *within each task* with replacement, computes a score for each seed/run,
    averages across seeds per task, and then averages across tasks to get a "mean" score.

    Metrics are selected per task using `task2metric`, with implementations stored in `metrics`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table with at least these columns:
        - task        : str   task name
        - prediction  : str   model prediction (stringified; cast internally)
        - label       : str   gold label (stringified; cast internally)

        Optional:
        - seed : Union[int, str]
            Run identifier. If missing, the function adds `seed = 0`.

        Optional (needed for NER-style evaluation):
        - text : str
            Sentence tokens (space-separated) corresponding to the BIO label sequences,
            used to reconstruct spans for NER evaluation.

        Notes on expected formats:
        - Classification tasks typically store scalar labels/predictions.
        - Multi-label tasks store comma-separated labels (e.g., "premise,claim" or "O").
        - NER tasks (task starts with "ner_" or equals "germeval_opinions") store BIO tag
          sequences as space-separated strings in `label` and `prediction`, and token
          sequences as a space-separated string in `text`.

    n_samples : int, default=1000
        Number of bootstrap resamples.

    seed : int, default=42
        Random seed used to make the bootstrap process reproducible. Each bootstrap
        iteration uses `seed + sample_idx` as its sampling RNG.

    Returns
    -------
    dict
        Dictionary with:
        - one key per task: the average (across seeds) score on the *original* data
        - 'mean': arithmetic mean across tasks (on the original data)
        - 'bootstrap': list of length `n_samples`, each entry is a dict:
            {'mean': <boot_mean>, <task1>: <boot_task1>, ...}

    Raises
    ------
    ValueError
        - If `task2metric` does not define a metric for some task in `df`.
        - If the internal train/metric special-cases hit unsupported formats.

    - The function prints per-task instance counts and the final non-bootstrapped scores.

    Examples
    --------
    >>> df = aggregate("/path/to/exp")
    >>> res = bootstrap_supergleber_scores(df, n_samples=2000, seed=123)
    >>> res["mean"], res["bootstrap"][0]["mean"]
    """

    np.random.seed(seed)  # Set random seed for reproducibility

    # Precompute unique tasks and seeds
    unique_tasks = df['task'].unique()
    if 'seed' not in df:
        df['seed'] = 0
    unique_seeds = df['seed'].unique()


    def compute_metric(task, correct, predicted, text):
        try:
            metric_key = task2metric[task]
        except KeyError:
            raise ValueError(f"No metric defined for task {task}")

        if task in ['argument_mining', 'db_aspect', 'hotel_aspect', 'up_dep', 'up_pos']:
            # multi-label F1 (macro) computation

            # Split multi-label strings into sets
            y_true = [set(l.split(',')) for l in correct]
            y_pred = [set(p.split(',')) if isinstance(p, str) else set() for p in predicted]

            # Build label vocabulary
            all_labels = sorted(set.union(*y_true, *y_pred))
            if 'O' in all_labels:
                all_labels.remove('O')
                y_pred = [labels if 'O' not in labels else set() for labels in y_pred]
                y_true = [labels if 'O' not in labels else set() for labels in y_true]

            label_to_idx = {l: i for i, l in enumerate(all_labels)}

            # Convert to multi-hot arrays
            y_true_bin = np.zeros((len(y_true), len(all_labels)), dtype=int)
            y_pred_bin = np.zeros((len(y_pred), len(all_labels)), dtype=int)

            for i, labels in enumerate(y_true):
                for l in labels:
                    y_true_bin[i, label_to_idx[l]] = 1
            for i, labels in enumerate(y_pred):
                for l in labels:
                    y_pred_bin[i, label_to_idx[l]] = 1

            # Compute macro F1
            score = metrics[metric_key](y_true_bin, y_pred_bin, zero_division=0)
            return score
        elif task.startswith('ner_') or task == 'germeval_opinions':
            DEBUG = False

            def bio_to_spans(seq: Iterable[Tuple[str, str]]) -> List[Tuple[str, str, int, int]]:
                """
                Convert token-level BIO labels to spans (Flair-compatible).
                Returns list of (span_text, label, start_idx, end_idx_exclusive)
                """
                toks = [t for t, _ in seq]
                labs = [l for _, l in seq]
                spans = []
                current_label = None
                start = 0

                for i, lab in enumerate(labs):
                    if lab in ("O", "o"):
                        if current_label is not None:
                            spans.append((" ".join(toks[start:i]), current_label, start, i))
                            current_label = None
                    elif lab.startswith(("B-", "b-")):
                        if current_label is not None:
                            spans.append((" ".join(toks[start:i]), current_label, start, i))
                        current_label = lab.split("-", 1)[1]
                        start = i
                    elif lab.startswith(("I-", "i-")):
                        type_ = lab.split("-", 1)[1]
                        if current_label != type_:
                            # treat as B-
                            if current_label is not None:
                                spans.append((" ".join(toks[start:i]), current_label, start, i))
                            current_label = type_
                            start = i
                    else:
                        # bare label: treat like B-
                        if current_label is not None:
                            spans.append((" ".join(toks[start:i]), current_label, start, i))
                        current_label = lab
                        start = i

                if current_label is not None:
                    spans.append((" ".join(toks[start:]), current_label, start, len(toks)))

                return spans

            def make_unlabeled_identifier(start, end, tokens, sentence_idx=None, style="flair_like"):
                """
                Create an identifier for a span matching Flair's internal style.

                Example: 1: Span[10:17]: "§ 14 Abs. 2 Satz 2 TzBfG"
                """
                # Flair uses 1-based token indices
                start_1b = start
                end_1b = end
                text = " ".join(tokens[start:end])

                if style == "flair_like":
                    prefix = f"{sentence_idx}: " if sentence_idx is not None else ""
                    return f'{prefix}Span[{start_1b}:{end_1b}]: "{text}"'

                # fallback: simple numeric form
                return f"{start_1b}:{end_1b}"

            def build_all_span_dicts_from_sentences(
                    sentences: List[List[Tuple[str, str, str]]],
                    id_style: str = "offset_text"
            ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str]]:
                """
                Build all_true_values and all_pred_values exactly as trainer does, but
                also return all_spans as a **list in insertion order** to guarantee deterministic iteration.
                """
                all_true_values: Dict[str, List[str]] = OrderedDict()
                all_pred_values: Dict[str, List[str]] = OrderedDict()
                all_spans_list: List[str] = []

                for sid, sentence in enumerate(sentences):
                    tokens = [t for (t, _, _) in sentence]
                    gold_seq = [(t, g) for (t, g, p) in sentence]
                    pred_seq = [(t, p) for (t, g, p) in sentence]

                    gold_spans = bio_to_spans(gold_seq)
                    pred_spans = bio_to_spans(pred_seq)

                    for span_text, label, start, end in gold_spans:
                        unlabeled = make_unlabeled_identifier(start, end, tokens)
                        representation = f"{sid}: {unlabeled}"
                        if representation not in all_true_values:
                            all_true_values[representation] = [label]
                        else:
                            all_true_values[representation].append(label)
                        if representation not in all_spans_list:
                            all_spans_list.append(representation)

                    for span_text, label, start, end in pred_spans:
                        unlabeled = make_unlabeled_identifier(start, end, tokens)
                        representation = f"{sid}: {unlabeled}"
                        if representation not in all_pred_values:
                            all_pred_values[representation] = [label]
                        else:
                            all_pred_values[representation].append(label)
                        if representation not in all_spans_list:
                            all_spans_list.append(representation)

                if DEBUG:
                    print("DEBUG span counts (ordered):",
                          "gold =", len(all_true_values),
                          "pred =", len(all_pred_values),
                          "union =", len(all_spans_list))
                    print("  example spans:", all_spans_list[:10])

                return all_true_values, all_pred_values, all_spans_list

            def align_span_lists_like_trainer(
                    all_spans_ordered: Sequence[str],
                    all_true_values: Dict[str, List[str]],
                    all_predicted_values: Dict[str, List[str]],
                    exclude_labels: Optional[List[str]] = None,
            ):
                """
                Exact alignment the trainer uses, iterating spans in the given order.
                Returns two lists true_values_span_aligned, predicted_values_span_aligned.
                """
                exclude_labels = set(exclude_labels or [])
                true_values_span_aligned = []
                predicted_values_span_aligned = []

                for span in all_spans_ordered:
                    list_of_gold = list(all_true_values.get(span, ["O"]))  # default ["O"] like trainer
                    # delete excluded labels
                    list_of_gold = [g for g in list_of_gold if g not in exclude_labels]
                    if not list_of_gold:
                        # trainer continues (skips)
                        continue
                    true_values_span_aligned.append(list_of_gold)
                    predicted_values_span_aligned.append(all_predicted_values.get(span, ["O"]))

                if DEBUG:
                    print("DEBUG aligned instances:", len(true_values_span_aligned))
                return true_values_span_aligned, predicted_values_span_aligned

            def evaluate_like_trainer(
                    true_aligned: List[List[str]],
                    pred_aligned: List[List[str]],
                    all_true_values: Dict[str, List[str]],
                    all_predicted_values: Dict[str, List[str]],
            ) -> Tuple[str, Dict[str, Any], Dict[Any, Any], str]:
                """
                Fully reproduce the trainer's sklearn evaluation logic.
                Returns classification_report_str, classification_report_dict, scores_dict, summary_str
                """
                # Build evaluation_label_dictionary equivalent (we simulate with OrderedDict mapping -> index)
                eval_label_list = ["O"]
                eval_map = {"O": 0}
                # iterate in trainer order: all_true_values.values(), then all_predicted_values.values()
                for true_values in all_true_values.values():
                    for lab in true_values:
                        if lab not in eval_map:
                            eval_map[lab] = len(eval_label_list)
                            eval_label_list.append(lab)
                for pred_values in all_predicted_values.values():
                    for lab in pred_values:
                        if lab not in eval_map:
                            eval_map[lab] = len(eval_label_list)
                            eval_label_list.append(lab)

                # detect multi-label
                multi_label = False
                for t, p in zip(true_aligned, pred_aligned):
                    if len(t) > 1 or len(p) > 1:
                        multi_label = True
                        break

                # Build y_true / y_pred exactly like trainer
                if multi_label:
                    y_true = []
                    y_pred = []
                    for true_instance in true_aligned:
                        arr = np.zeros(len(eval_label_list), dtype=int)
                        for val in true_instance:
                            arr[eval_map[val]] = 1
                        y_true.append(arr.tolist())
                    for pred_instance in pred_aligned:
                        arr = np.zeros(len(eval_label_list), dtype=int)
                        for val in pred_instance:
                            arr[eval_map[val]] = 1
                        y_pred.append(arr.tolist())
                else:
                    y_true = [eval_map[inst[0]] for inst in true_aligned]
                    y_pred = [eval_map[inst[0]] for inst in pred_aligned]

                # Prepare target_names & labels using same counter logic
                counter = Counter(itertools.chain.from_iterable(all_true_values.values()))
                counter.update(list(itertools.chain.from_iterable(all_predicted_values.values())))

                target_names = []
                labels_indices = []
                for label_name, _count in counter.most_common():
                    if label_name == "O":
                        continue
                    if label_name not in eval_map:
                        continue
                    target_names.append(label_name)
                    labels_indices.append(eval_map[label_name])

                # compute sklearn reports
                if len(all_true_values) + len(all_predicted_values) > 1:
                    cr_str = sklearn.metrics.classification_report(
                        y_true, y_pred, digits=4, target_names=target_names, zero_division=0, labels=labels_indices
                    )
                    cr_dict = sklearn.metrics.classification_report(
                        y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True,
                        labels=labels_indices
                    )

                    accuracy_score_val = round(sklearn.metrics.accuracy_score(y_true, y_pred), 4)

                    if len(target_names) == 1:
                        cr_dict["micro avg"] = cr_dict["macro avg"]

                    if "micro avg" not in cr_dict:
                        cr_dict["micro avg"] = {}
                        for precision_recall_f1 in cr_dict["macro avg"]:
                            cr_dict["micro avg"][precision_recall_f1] = cr_dict["accuracy"]

                    detailed_result = (
                            "\nResults:"
                            f"\n- F-score (micro) {round(cr_dict['micro avg']['f1-score'], 4)}"
                            f"\n- F-score (macro) {round(cr_dict['macro avg']['f1-score'], 4)}"
                            f"\n- Accuracy {accuracy_score_val}"
                            "\n\nBy class:\n" + cr_str
                    )

                    scores = {}
                    for avg_type in ("micro avg", "macro avg"):
                        for metric_type in ("f1-score", "precision", "recall"):
                            scores[(avg_type, metric_type)] = cr_dict[avg_type][metric_type]
                    scores["accuracy"] = accuracy_score_val

                    return cr_str, cr_dict, scores, detailed_result
                else:
                    # fallback identical to trainer
                    return "", {}, {"loss": 0.0}, "No labels/predictions available"


            sentences = []

            for i in range(len(correct)):
                token_list = text.iloc[i]
                if isinstance(token_list, str):
                    token_list = token_list.split()
                else:
                    token_list = []
                gold_labels = correct.iloc[i].split()
                pred_labels = predicted.iloc[i].split()
                sentences.append(list(zip(token_list, gold_labels, pred_labels)))

            all_true_values, all_predicted_values, all_spans = build_all_span_dicts_from_sentences(sentences, id_style="offset_text")
            true_aligned, pred_aligned = align_span_lists_like_trainer(all_spans, all_true_values, all_predicted_values, exclude_labels=[])
            cr_str, cr_dict, scores_dict, summary_str = evaluate_like_trainer(true_aligned, pred_aligned, all_true_values, all_predicted_values)

            return scores_dict[('micro avg', 'f1-score')]


        # ---- default behavior for other tasks ----
        try:
            metric = metrics[metric_key]
        except Exception as e:
            raise e

        if task != 'similarity_pawsx':
            encoder = LabelEncoder()
            preds = predicted.tolist()
            refs = correct.tolist()
            encoder.fit(preds + refs)  # include all labels
            preds_enc = encoder.transform(preds)
            refs_enc = encoder.transform(refs)
            if metric_key in ['accuracy']:
                result = metric(predictions=preds_enc, references=refs_enc)
            else:
                result = metric(preds_enc, refs_enc)

        else:
            preds_enc = [float(p) for p in predicted]
            refs_enc = [float(p) for p in correct]
            result = metric(predictions=preds_enc, references=refs_enc)

        if isinstance(result, float):
            return result

        metric_key_short = metric_key.removeprefix('micro_').removeprefix('macro_')
        return result[metric_key_short]

    if 'idx' not in df:
        df = df.reset_index().rename(columns={'index': 'idx'})

    df['prediction'] = df['prediction'].astype(str)
    df['label'] = df['label'].astype(str)

    # Prepare the pivot table once for efficiency
    pivot_tables = {
        task: df[df['task'] == task]
        .pivot_table(index='idx', columns='seed', values='prediction', aggfunc='first')
        .merge(df[df['task'] == task][['idx', 'label', 'text']].drop_duplicates(), on='idx')
        for task in unique_tasks
    }

    # Predefine bootstrap scores list
    bootstrap_scores = []

    mean_tasks = unique_tasks

    # Bootstrap loop
    for sample_idx in tqdm(range(n_samples), desc="Bootstrapping"):
        task_scores = {}

        for task in unique_tasks:
            task_pivot_df = pivot_tables[task]

            # Bootstrap sample
            sampled_pivot_df = task_pivot_df.sample(
                n=len(task_pivot_df), replace=True, random_state=seed + sample_idx
            )

            # Compute scores for each seed
            scores = [
                compute_metric(task, sampled_pivot_df['label'], sampled_pivot_df[seed], sampled_pivot_df['text'])
                for seed in unique_seeds
            ]

            if 0.0 in scores:
                print(f"Sample {sample_idx} has 0.0 in scores for task {task}")
                #raise ValueError(f"Sample {sample_idx} has 0.0 in scores for task {task}")

            task_scores[task] = np.mean(scores)

        bootstrap_scores.append({'mean': np.mean([task_scores[task] for task in mean_tasks]), **task_scores})

    # compute the mean of the original data
    result = {}
    for task in unique_tasks:
        task_pivot_df = pivot_tables[task]
        scores = [
            compute_metric(task, task_pivot_df['label'], task_pivot_df[seed], task_pivot_df['text'])
            for seed in unique_seeds
        ]

        result[task] = np.mean(scores)

    print(json.dumps(df.groupby('task').size().to_dict(), indent=2))
    result['mean'] = np.mean([result[task] for task in mean_tasks])
    print(json.dumps(result, indent=2))

    result['bootstrap'] = bootstrap_scores

    return result


def supergleber_bootstrap(input, n_samples=1000, confidence_level=0.95, suffix='', seeds=None):
    """
    Run (or load) SuperGLEBer bootstrapping for an experiment directory, persist results,
    and compute confidence intervals for the aggregated mean score.

    This function is a convenience wrapper around:
    - `aggregate(...)` to build a unified predictions DataFrame (if not cached),
    - `bootstrap_supergleber_scores(...)` to generate bootstrap samples,
    - saving/loading the resulting JSON.

    Parameters
    ----------
    input : str
        Path to an experiment directory containing per-seed subfolders (see `aggregate`).
        The function stores outputs into this directory.

    n_samples : int, default=1000
        Number of bootstrap samples.

    confidence_level : float, default=0.95
        Confidence level used for a symmetric normal-approximation CI around the bootstrap mean.
        Also prints percentile-based 95% and 99% intervals regardless of this parameter.

    suffix : str, default=''
        Suffix used to distinguish different result variants on disk.
        - Cached per-example CSV:   f"{input}/results{suffix}.csv"
        - Bootstrapped JSON:        f"{input}/results_{n_samples}_{confidence_level}{suffix}.json"

    seeds : Optional[Iterable[Union[int, str]]], default=None
        If provided, restricts aggregation to specific seed folders (passed through to `aggregate`).

    Returns
    -------
    dict
        The dictionary returned by `bootstrap_supergleber_scores`, augmented with:
        - 'bootstrap_mean' : float  mean of bootstrapped 'mean' scores
        - 'ci_lower'       : float  symmetric normal-approx lower bound
        - 'ci_upper'       : float  symmetric normal-approx upper bound
        - 'margin_of_error': float  half-width of the symmetric CI

    Side Effects
    ------------
    - Writes/reads cached files inside `input/`.
    - Prints:
      - first 10 bootstrapped mean scores
      - mean bootstrapped score
      - percentile CIs (95% and 99%)
      - symmetric normal-approx CI
      - original (non-bootstrapped) mean score

    Notes
    -----
    Percentile intervals are computed directly from the bootstrap distribution of 'mean'.
    The symmetric CI uses a normal approximation:
        mean ± z_(alpha/2) * std
    where std is the sample standard deviation of bootstrap means.
    """

    output = f"{input}/results_{n_samples}_{confidence_level}{suffix}.json"

    tasks = all_supergleber_tasks

    if os.path.exists(output):
        with open(output, "r") as f:
            bootstrap_results = json.load(f)
    else:
        df_output = f'{input}/results{suffix}.csv'
        if os.path.exists(df_output):
            df = pd.read_csv(df_output)
        else:
            df = aggregate(input, tasks=tasks, seeds=seeds)
            df.to_csv(df_output, index=False)

        # Run bootstrapping
        print(f"Running bootstrapping on SuperGLEBer scores for {input}...")
        bootstrap_results = bootstrap_supergleber_scores(df, n_samples=n_samples, seed=42)

        # Save bootstrapped scores to a JSON file
        with open(output, "w") as f:
            json.dump(bootstrap_results, f)

    mean_bootstrap_results = [result['mean'] for result in bootstrap_results['bootstrap']]
    print(f"Bootstrapped GLUE scores (first 10): {mean_bootstrap_results[:10]}")
    print(f"Mean Bootstrapped GLUE score: {np.mean(mean_bootstrap_results):.2f}")
    # 0.95 and 0.99 confidence intervals
    print(f"95% CI: {np.percentile(mean_bootstrap_results, 2.5):.2f} - {np.percentile(mean_bootstrap_results, 97.5):.2f}")
    print(f"99% CI: {np.percentile(mean_bootstrap_results, 0.5):.2f} - {np.percentile(mean_bootstrap_results, 99.5):.2f}")

    # Step 1: Compute the mean
    mean_score = np.mean(mean_bootstrap_results)

    # Step 2: Compute the standard error (SE)
    std_error = np.std(mean_bootstrap_results, ddof=1)  # Use ddof=1 for sample std dev
    margin_of_error = norm.ppf(1 - (1 - confidence_level) / 2) * std_error

    # Step 3: Calculate symmetric confidence intervals
    lower_bound = mean_score - margin_of_error
    upper_bound = mean_score + margin_of_error
    # Print results
    print(f"Mean Bootstrapped GLUE score: {mean_score:.4f}")
    print(f"Symmetric 95% CI: {lower_bound:.2f} - {upper_bound:.2f}")
    print(f"Mean Bootstrapped GLUE score: {mean_score:.4f} ± {margin_of_error:.4f}")

    print(f"Normal Mean (without bootstrapping): {bootstrap_results['mean']:.4f}")

    bootstrap_results['bootstrap_mean'] = mean_score
    bootstrap_results['ci_lower'] = lower_bound
    bootstrap_results['ci_upper'] = upper_bound
    bootstrap_results['margin_of_error'] = margin_of_error

    return bootstrap_results

def non_bootstrap_glue_scores(input):
    # return normal scores, including mean without bootstrapping
    df_output = f'{input}/results.csv'
    if os.path.exists(df_output):
        df = pd.read_csv(df_output)
    else:
        df = aggregate(input)
        df.to_csv(df_output, index=False)
    tasks = all_supergleber_tasks
    result = {}
    for task in tasks:
        task_df = df[df['task'] == task]
        seeds = task_df['seed'].unique()
        scores = []
        for seed in seeds:
            preds = task_df[task_df['seed'] == seed]['prediction']
            labels = task_df[task_df['seed'] == seed]['label']
            score = bootstrap_supergleber_scores(pd.DataFrame({
                'task': [task]*len(labels),
                'seed': [seed]*len(labels),
                'prediction': preds,
                'label': labels
            }), n_samples=1, seed=42)[0][task]
            scores.append(score)
        result[task] = np.mean(scores)
    result['mean'] = np.mean([result[task] for task in tasks])
    return result


def glue_bootstrap_scores(input, n_samples=1000, confidence_level=0.95):
    bootstrap_results = supergleber_bootstrap(input, n_samples, confidence_level)

    if isinstance(bootstrap_results, dict):
        bootstrap_results = [x['mean'] for x in bootstrap_results['bootstrap']]

    # Step 1: Compute the mean
    mean_score = np.mean(bootstrap_results)

    # Step 2: Compute the standard error (SE)
    std_error = np.std(bootstrap_results, ddof=1)  # Use ddof=1 for sample std dev
    margin_of_error = norm.ppf(1 - (1 - confidence_level) / 2) * std_error

    return mean_score, margin_of_error

if __name__ == '__main__':
    base_folder = 'outputs/bert-base-german-cased'
    mean, margin = glue_bootstrap_scores(base_folder, n_samples=3, confidence_level=0.95)
    print(f"Final Result: {mean:.4f} ± {margin:.4f}")
