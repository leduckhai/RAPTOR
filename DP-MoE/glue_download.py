#!/usr/bin/env python3
"""
Download GLUE tasks: SST-2, MNLI, QNLI, QQP
Saves to: {data_root}/GLUE-{TASK}/train.tsv + dev.tsv

Usage:
    python download_glue.py --data_root /home/jovyan/gpus-4-nodes-volume/duc/data
    python download_glue.py --data_root /home/jovyan/gpus-4-nodes-volume/duc/data --tasks sst2 mnli
"""
import argparse, os, csv, sys

def download_from_hf(task: str, out_dir: str) -> None:
    """Download via HuggingFace datasets library (most reliable)."""
    from datasets import load_dataset

    HF_NAMES = {
        "sst2": ("glue", "sst2",   {"train":"train", "dev":"validation"}),
        "mnli": ("glue", "mnli",   {"train":"train", "dev":"validation_matched"}),
        "qnli": ("glue", "qnli",   {"train":"train", "dev":"validation"}),
        "qqp":  ("glue", "qqp",    {"train":"train", "dev":"validation"}),
    }

    # Column mapping: HF field name → TSV column name expected by load_glue()
    COL_MAP = {
        "sst2": {"sentence":"sentence", "label":"label"},
        "mnli": {"premise":"premise", "hypothesis":"hypothesis", "label":"gold_label",
                 "label_names": {0:"entailment", 1:"neutral", 2:"contradiction"}},
        "qnli": {"question":"question", "sentence":"sentence", "label":"label",
                 "label_names": {0:"entailment", 1:"not_entailment"}},
        "qqp":  {"question1":"question1", "question2":"question2", "label":"label"},
    }

    hf_path, hf_name, split_map = HF_NAMES[task]
    print(f"[{task}] Downloading from HuggingFace ({hf_path}/{hf_name})...")
    ds = load_dataset(hf_path, hf_name)
    cm = COL_MAP[task]

    os.makedirs(out_dir, exist_ok=True)

    for split_name, tsv_name in [("train","train.tsv"), ("dev", "dev_matched.tsv" if task=="mnli" else "dev.tsv")]:
        hf_split = split_map.get(split_name if split_name=="train" else "dev")
        data = ds[hf_split]
        out_path = os.path.join(out_dir, tsv_name)

        label_names = cm.get("label_names", {})

        with open(out_path, "w", encoding="utf-8", newline="") as f:
            if task == "sst2":
                cols = ["sentence", "label"]
                writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
                writer.writeheader()
                for row in data:
                    writer.writerow({"sentence": row["sentence"], "label": row["label"]})

            elif task == "mnli":
                cols = ["premise", "hypothesis", "gold_label"]
                writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
                writer.writeheader()
                for row in data:
                    lname = label_names.get(row["label"], str(row["label"]))
                    writer.writerow({"premise": row["premise"],
                                     "hypothesis": row["hypothesis"],
                                     "gold_label": lname})

            elif task == "qnli":
                cols = ["question", "sentence", "label"]
                writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
                writer.writeheader()
                for row in data:
                    lname = label_names.get(row["label"], str(row["label"]))
                    writer.writerow({"question": row["question"],
                                     "sentence": row["sentence"],
                                     "label": lname})

            elif task == "qqp":
                cols = ["question1", "question2", "label"]
                writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
                writer.writeheader()
                for row in data:
                    writer.writerow({"question1": row["question1"],
                                     "question2": row["question2"],
                                     "label": row["label"]})

        n = len(data)
        print(f"  [{task}] {split_name} → {out_path}  ({n:,} rows)")

    print(f"[{task}] Done → {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Download GLUE tasks as TSV files")
    p.add_argument("--data_root", default="/home/jovyan/gpus-4-nodes-volume/duc/data",
                   help="Root directory. Each task saved to {data_root}/GLUE-{TASK}/")
    p.add_argument("--tasks", nargs="+", default=["sst2","mnli","qnli","qqp"],
                   choices=["sst2","mnli","qnli","qqp"],
                   help="Which tasks to download")
    args = p.parse_args()

    try:
        import datasets
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    TASK_DIRS = {
        "sst2": "GLUE-SST-2",
        "mnli": "GLUE-MNLI",
        "qnli": "GLUE-QNLI",
        "qqp":  "GLUE-QQP",
    }

    print(f"Downloading tasks: {args.tasks}")
    print(f"Root: {args.data_root}\n")

    for task in args.tasks:
        out_dir = os.path.join(args.data_root, TASK_DIRS[task])
        train_exists = os.path.exists(os.path.join(out_dir, "train.tsv"))
        dev_file = "dev_matched.tsv" if task == "mnli" else "dev.tsv"
        dev_exists = os.path.exists(os.path.join(out_dir, dev_file))

        if train_exists and dev_exists:
            # Count lines
            with open(os.path.join(out_dir,"train.tsv")) as f:
                n = sum(1 for _ in f) - 1
            print(f"[{task}] Already exists ({n:,} train rows) → {out_dir}  SKIP")
            continue

        try:
            download_from_hf(task, out_dir)
        except Exception as e:
            print(f"[{task}] ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\nAll done. Verify with:")
    for task in args.tasks:
        d = os.path.join(args.data_root, TASK_DIRS[task])
        dev_f = "dev_matched.tsv" if task=="mnli" else "dev.tsv"
        for f in ["train.tsv", dev_f]:
            path = os.path.join(d, f)
            if os.path.exists(path):
                with open(path) as fp: n = sum(1 for _ in fp) - 1
                print(f"  OK  {path}  ({n:,} rows)")
            else:
                print(f"  MISSING  {path}")


if __name__ == "__main__":
    main()