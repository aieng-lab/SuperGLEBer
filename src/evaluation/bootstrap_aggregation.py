import os
import pandas as pd
import re


task2train_type = {
    'germanquad': 'hf_qa',
    'mlqa': 'hf_qa',
    'similarity_pawsx': 'sentence_transformers',
    'up_dep': 'flair_plain',
    'up_pos': 'flair_plain',
    'ner_biofid': 'flair_plain_subgroups2',
    'ner_europarl': 'flair_plain_subgroups2',
    'ner_legal': 'flair_plain_subgroups2',
    'ner_news': 'flair_plain_subgroups2',
    'ner_wiki_news': 'flair_plain_subgroups2',
    'germeval_opinions': 'flair_plain_subgroups2',
    'massive_seq': 'flair_plain',
    'argument_mining': 'flair_multi_labels',
    'db_aspect': 'flair_multi_labels',
    'hotel_aspect': 'flair_multi_labels',
}


ALL_SUPER_GLEBER_TASKS = (
    'webcage',
    'verbal_idioms',
    'up_pos',
    'up_dep',
    'toxic_comments',
    'topic_relevance',
    'similarity_pawsx',
    'quest_ans',
    'query_ad',
    'polarity',
    'pawsx',
    'offensive_lang',
    'nli',
    'news_class',
    'ner_wiki_news',
    'ner_news',
    'ner_legal',
    'ner_europarl',
    'ner_biofid',
    'mlqa',
    'massive_seq',
    'massive_intents',
    'hotel_aspect',
    'germeval_opinions',
    'germanquad',
    'factclaiming_comments',
    'engaging_comments',
    'db_aspect',
    'argument_mining',
)

def aggregate(base_dir, tasks=ALL_SUPER_GLEBER_TASKS, seeds=None):
    """
    Aggregate per-task test-set predictions from multiple training runs (seeds) into
    one unified Pandas DataFrame.

    This function expects a directory layout like:

        base_dir/
            <seed>/
                <task_name>/
                    training_logs/
                        ... task-specific prediction files ...

    where `<seed>` is an integer folder name (e.g., "0", "1", "2"), and each `<task_name>`
    is one of the entries in `tasks`.

    The exact prediction file that gets read depends on the task's "train type", which is
    looked up via `task2train_type`:

    - "flair"                  -> parses `training_logs/test.tsv` as block-structured text
    - "flair_multi_labels"     -> parses `training_logs/test.tsv` as block-structured text,
                                  includes an additional `example_id` during parsing (then dropped)
    - "flair_plain"            -> reads `training_logs/test.tsv` as whitespace-separated columns
    - "flair_plain_subgroups"  -> like flair_plain, then keeps only entity-start labels (drops "O"
                                  and labels starting with "I-")
    - "flair_plain_subgroups2" -> reads CoNLL-like token-level files and aggregates into one row
                                  per sentence (space-joined tokens/labels/preds)
    - "hf_qa"                  -> reads `training_logs/results.csv` and maps columns:
                                  prediction <- pred_label, label <- true_label, text <- ""
    - "sentence_transformers"  -> reads `training_logs/predictions.csv`, filters out rows with
                                  true_label == '-', and sets:
                                  text <- sentence1 + " ||| " + sentence2
                                  label <- float(true_label)
                                  prediction <- float(predicted_score)

    Parameters
    ----------
    base_dir : str
        Path to the directory that contains per-seed subfolders (integer names).
        Each seed folder must contain one subfolder per task in `tasks`.

    tasks : Iterable[str], default=ALL_SUPER_GLEBER_TASKS
        Iterable of task names to process for each seed folder. For each seed folder,
        the function expects `os.path.join(base_dir, seed, task_name)` to exist.
        If a task folder is missing, a ValueError is raised.

    seeds : Optional[Iterable[Union[int, str]]], default=None
        If provided, restrict aggregation to only these seed folder names.
        Seed values are stringified (e.g., [0, 1] -> ["0", "1"]) and matched against the
        immediate children of `base_dir`.

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame with one row per evaluated example (task-dependent granularity),
        containing the columns:

        - text        : str
        - prediction  : Union[str, float]
        - label       : Union[str, float]
        - task        : str   (task name)
        - seed        : str   (seed folder name)

        Notes on granularity:
        - For sequence labeling in "flair_plain_subgroups2", rows are sentence-level.
        - For "flair_plain" / "flair_plain_subgroups", rows are line-based as stored in the TSV.
        - For QA / sentence-transformers, rows are dataset-example-level as in the CSVs.

    Raises
    ------
    ValueError
        - If a required task folder does not exist for a given seed.
        - If a task has an unknown train type (not handled by the function).

    Side Effects
    ------------
    Prints progress information to stdout:
    - which seeds are used (if `seeds` is provided),
    - skipped non-integer folders,
    - currently processed seed folder.

    File Expectations (by train type)
    ---------------------------------
    flair:
        <task_folder>/training_logs/test.tsv
        Expected to contain blank-line-separated blocks, each with:
        - first line: "<text> <id>"
        - lines containing "- Gold:" and "- Pred:"

    flair_multi_labels:
        Same as flair, but gold labels may be a comma-separated list. Gold labels are normalized:
        - empty -> "O"
        - otherwise comma-joined list

    flair_plain / flair_plain_subgroups:
        <task_folder>/training_logs/test.tsv
        Whitespace-separated columns: text, label, prediction (read with `pd.read_csv(..., sep=' ')`).
        For flair_plain_subgroups, rows with label == "O" or label starting with "I-" are removed.

    flair_plain_subgroups2:
        <task_folder>/training_logs/test.tsv
        CoNLL-like token lines separated by blank lines. Each non-empty line may contain:
        - 1 column: token
        - 2 columns: token gold
        - 3+ columns: token gold pred
        The function aggregates each sentence into:
        - text: tokens joined with spaces
        - label: gold tags joined with spaces
        - prediction: predicted tags joined with spaces

    hf_qa:
        <task_folder>/training_logs/results.csv
        Must contain `pred_label` and `true_label` columns.

    sentence_transformers:
        <task_folder>/training_logs/predictions.csv
        Must contain columns: `sentence1`, `sentence2`, `true_label`, `predicted_score`.
        Rows with true_label == '-' are dropped.

    Examples
    --------
    Aggregate all default tasks for all integer seed folders under `./runs`:

    >>> df = aggregate("./runs")
    >>> df.head()

    Aggregate only selected tasks and only seeds 0 and 3:

    >>> df = aggregate("./runs", tasks=["germanquad", "mlqa"], seeds=[0, 3])
    >>> df["task"].value_counts()
    """

    result = []

    column_names = ['text', 'label', 'prediction']
    folders = os.listdir(base_dir)
    if seeds is not None:
        seeds = [str(s) for s in seeds]
        folders = [f for f in folders if f in seeds]
        print(f"Using specified seeds only: {folders}")

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # only use folders that are integers
        if not folder.isdigit():
            print(f"Skipping non-integer folder: {folder_path}")
            continue

        print(f'Processing folder: {folder_path}')
        for task_name in tasks:
            task_folder = os.path.join(folder_path, task_name)
            if not os.path.isdir(task_folder):
                raise ValueError(f"No task folder found in {task_folder}")

            train_type = task2train_type.get(task_name, 'flair')
            if train_type == 'flair':
                test_file = os.path.join(task_folder, 'training_logs', 'test.tsv')
                # Read file
                with open(test_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                # Split into blocks by blank lines
                blocks = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]

                rows = []
                for block in blocks:
                    # Extract text and ID (ID at end of first line)
                    first_line = block.split("\n")[0].strip()
                    text_match = re.match(r"^(.*)\s+\d+$", first_line)
                    text = text_match.group(1) if text_match else first_line

                    # Extract Gold and Pred
                    gold_match = re.search(r"- Gold:\s*(.*)", block)
                    pred_match = re.search(r"- Pred:\s*(.*)", block)
                    gold = gold_match.group(1).strip()
                    pred = pred_match.group(1).strip()

                    rows.append({"text": text, "label": gold, "prediction": pred})
                df = pd.DataFrame(rows, columns=column_names)

            elif train_type == 'flair_multi_labels':
                test_file = os.path.join(task_folder, 'training_logs', 'test.tsv')
                # Read file
                with open(test_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                # Split into blocks by blank lines (robust to whitespace-only blank lines)
                blocks = [b.strip() for b in re.split(r"(?:\r?\n\s*\r?\n)+", content) if b.strip()]

                rows = []
                for block in blocks:
                    # Extract text and ID (ID at end of first line)
                    first_line = block.split("\n")[0].strip()
                    m = re.match(r"^(.*)\s+(\d+)\s*$", first_line)
                    if m:
                        text = m.group(1).strip()
                        example_id = m.group(2)
                    else:
                        text = first_line
                        example_id = None

                    # Extract Gold and Pred
                    gold_match = re.search(r"- Gold:\s*(.*)", block)
                    pred_match = re.search(r"- Pred:\s*(.*)", block)
                    gold = gold_match.group(1).strip() if gold_match else ""
                    pred = pred_match.group(1).strip() if pred_match else ""

                    # normalize gold list to comma-joined string (one row per span)
                    if ',' in gold:
                        golds = [g.strip() for g in gold.split(',') if g.strip()]
                    elif gold.strip() == "":
                        golds = []
                    else:
                        golds = [gold.strip()]

                    gold_cell = ",".join(golds) if golds else "O"

                    rows.append({"text": text, "label": gold_cell, "prediction": pred, "example_id": example_id})

                df = pd.DataFrame(rows, columns=(column_names + ['example_id']))
            elif train_type == 'flair_plain_subgroups':
                test_file = os.path.join(task_folder, 'training_logs', 'test.tsv')
                df = pd.read_csv(test_file, header=None, sep=' ', on_bad_lines='skip', quoting=3, names=column_names)
                df = df[(df['label'] != 'O') & ~df['label'].str.startswith('I-')].reset_index(drop=True)

            elif train_type == 'flair_plain_subgroups2':
                test_file = os.path.join(task_folder, "training_logs", "test.tsv")
                sentences = []
                cur = []

                with open(test_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if not line.strip():
                            if cur:
                                sentences.append(cur)
                                cur = []
                            continue

                        parts = line.split()
                        # tolerate 1–3 columns
                        if len(parts) == 1:
                            token = parts[0]
                            gold = "O"
                            pred = "O"
                        elif len(parts) == 2:
                            token, gold = parts
                            pred = "O"
                        else:
                            token, gold, pred = parts[0], parts[1], parts[2]

                        cur.append((token, gold, pred))

                if cur:
                    sentences.append(cur)

                # aggregate into one row per sentence
                dict_data = {"text": [], "label": [], "prediction": []}
                for sent in sentences:
                    tokens, golds, preds = zip(*sent)
                    dict_data["text"].append(" ".join(tokens))
                    dict_data["label"].append(" ".join(golds))
                    dict_data["prediction"].append(" ".join(preds))

                df = pd.DataFrame(dict_data)
            elif train_type == 'flair_plain':
                test_file = os.path.join(task_folder, 'training_logs', 'test.tsv')
                df = pd.read_csv(test_file, header=None, sep=' ', on_bad_lines='skip', quoting=3, names=column_names)
            elif train_type == 'hf_qa':
                test_file = os.path.join(task_folder, 'training_logs', 'results.csv')
                df = pd.read_csv(test_file)
                df['text'] = ""
                df['prediction'] = df['pred_label']
                df['label'] = df['true_label']
            elif train_type == 'sentence_transformers':
                test_file = os.path.join(task_folder, 'training_logs', 'predictions.csv')
                df = pd.read_csv(test_file)
                df = df[df['true_label'] != '-'].reset_index(drop=True)
                df['label'] = df['true_label'].astype(float)
                df['prediction'] = df['predicted_score'].astype(float)
                df['text'] = df['sentence1'] + " ||| " + df['sentence2']
            else:
                raise ValueError(f"Unknown train type {train_type} for task {task_name}")

            # drop other columns
            df = df[['text', 'prediction', 'label']]
            df['task'] = task_name
            df['seed'] = folder
            result.append(df)

    final_df = pd.concat(result, ignore_index=True)
    return final_df
