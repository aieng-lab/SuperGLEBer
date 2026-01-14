# SuperGLEBer with Bootstrapping

This repository is a SuperGLEBer variant adapted for the evaluation of [Understanding or Memorizing? A Case Study of German Definite Articles
in Language Models](todo).
See [https://github.com/aieng-lab/gradiend-german-articles](https://github.com/aieng-lab/gradiend-german-articles) for the main paper codebase.

Compared to upstream [SuperGLEBer](https://github.com/LSX-UniWue/SuperGLEBer), we keep the benchmark and training workflow largely intact. The key additions are:
- **Bootstrap confidence intervals** for all task metrics and the overall mean score (for more reliable comparisons).
- Minor training/config extensions (e.g., adding **EuroBERT-210m**).

> The default configuration (`src/conf/train_args/config.yml`) together with the provided `train_*.sh` scripts produces the required folder/file structure automatically.

## Reproducibility (GRADIEND paper)

### 1) Run experiments

Use the scripts in `shell_scripts/`:

- `shell_scripts/_config.sh`  
  Sets environment variables (e.g., task list, GRADIEND-specific model directory overrides).
- `shell_scripts/train.sh`  
  Runs all experiments used in the paper.
- `shell_scripts/train_{model}.sh`  
  Runs experiments for a specific model: 
  - `bert` ([`bert-base-german-cased`](https://huggingface.co/google-bert/bert-base-german-cased))
  - `euro` ([`EuroBERT/EuroBERT-210m`](https://huggingface.co/EuroBERT/EuroBERT-210m))
  - `gbert` ([deepset/gbert-large](https://huggingface.co/deepset/gbert-large))
  - `gpt` ([dbmdz/german.gpt2](https://huggingface.co/dbmdz/german-gpt2))
  - `modern` ([LSX-UniWue/ModernGBERT_1B](https://huggingface.co/LSX-UniWue/ModernGBERT_1B))
  - `llama` ([meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B))

Example:

```bash
bash shell_scripts/train.sh
```

### 2) Compute bootstrap confidence intervals and tables

Compute and print results (including bootstrap confidence intervals) via:
    
```bash
python src/evaluation/print_table.py
```


# ✨SuperGLEBer ✨

SuperGLEBer (German Language Understanding Evaluation Benchmark) is a broad Natural Language Understanding benchmark suite for the German language in order to create a better understanding of the current state of German LLMs.
Our benchmark consists of 29 different tasks ranging over different types like document classification, sequence tagging, sentence similarity, and question answering.

If you use this benchmark in your research, please cite the following paper:
<https://aclanthology.org/2024.naacl-long.438/>
For the current leaderboard and more information check out the [SuperGLEBer Website](https://supergleber.professor-x.de/) 🚀

This is the updated branch that contains the new and improved version of the SuperGLEBer benchmark.

## Running Experiments

create all relevant files necessary to schedule runs on a k8s/slurm cluster:

```bash
python src/template_k8s.py
```

running a model on a task:

```bash
python src/train.py +model=gbert_base +train_args=a100 +task=news_class
```

override config keys via CLI:

```bash
python src/train.py +model=gbert_base +train_args=a100 +task=news_class train_args.batch_size=1
```


pip install -r requirements.txt \
  --index-url https://download.pytorch.org/whl/cu121


you can find valid parameters in the provided yaml configs: <https://github.com/LSX-UniWue/SuperGLEBer/tree/paper/src/conf>
## Citation
```bib
@inproceedings{pfister-hotho-2024-supergleber,
    title = "{S}uper{GLEB}er: {G}erman Language Understanding Evaluation Benchmark",
    author = "Pfister, Jan  and
      Hotho, Andreas",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.438/",
    doi = "10.18653/v1/2024.naacl-long.438",
    pages = "7904--7923",
    abstract = "We assemble a broad Natural Language Understanding benchmark suite for the German language and consequently evaluate a wide array of existing German-capable models in order to create a better understanding of the current state of German LLMs. Our benchmark consists of 29 different tasks ranging over different types such as document classification, sequence tagging, sentence similarity, and question answering, on which we evaluate 10 different German-pretrained models, thereby charting the landscape of German LLMs. In our comprehensive evaluation we find that encoder models are a good choice for most tasks, but also that the largest encoder model does not necessarily perform best for all tasks. We make our benchmark suite and a leaderboard publically available at https://supergleber.professor-x.de and encourage the community to contribute new tasks and evaluate more models on it (https://github.com/LSX-UniWue/SuperGLEBer)."
}
```
