import argparse
import os
import warnings

import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from tqdm import tqdm

from ESA.annotation_loader import AnnotationLoader


warnings.filterwarnings("ignore")

SEVERITY_TO_LABEL = {
    "minor": 1,
    "major": 2,
    "critical": 3,
}
IGNORED_SEVERITIES = {"undecided"}
NO_ERROR_LABEL = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract xCOMET all-layer token representations with multilabel severity tags."
    )
    parser.add_argument("--dataset", type=str, default="ESA-1", help="Protocol name to load.")
    parser.add_argument(
        "--model",
        type=str,
        default="Unbabel/XCOMET-XL",
        help="COMET model checkpoint name.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="extract_multilabel_output",
        help="Directory for extracted tensors and metadata.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="experiment_esa1_all_layers_multilabel.pt",
        help="Output PT filename.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Optional cap on the number of usable samples. -1 means all.",
    )
    return parser.parse_args()


def has_usable_spans(spans):
    if not isinstance(spans, list):
        return False

    for span in spans:
        severity = str(span.get("severity", "")).lower()
        if severity in IGNORED_SEVERITIES:
            continue
        if severity not in SEVERITY_TO_LABEL:
            continue

        start_i = span.get("start_i")
        end_i = span.get("end_i")
        if str(start_i).isdigit() and str(end_i).isdigit():
            return True

    return False


def filter_supported_spans(spans):
    filtered = []
    if not isinstance(spans, list):
        return filtered

    for span in spans:
        severity = str(span.get("severity", "")).lower()
        if severity in IGNORED_SEVERITIES:
            continue
        if severity not in SEVERITY_TO_LABEL:
            continue

        start_i = span.get("start_i")
        end_i = span.get("end_i")
        if not (str(start_i).isdigit() and str(end_i).isdigit()):
            continue

        filtered.append(
            {
                "start_i": int(start_i),
                "end_i": int(end_i),
                "severity": severity,
            }
        )

    return filtered


def get_token_labels(spans, offsets):
    token_labels = [NO_ERROR_LABEL] * len(offsets)

    for token_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue

        max_label = NO_ERROR_LABEL
        for span in spans:
            if max(span["start_i"], tok_start) < min(span["end_i"], tok_end):
                max_label = max(max_label, SEVERITY_TO_LABEL[span["severity"]])

        token_labels[token_idx] = max_label

    return token_labels


def find_input_dict(obj):
    if isinstance(obj, dict) and "input_ids" in obj:
        return obj
    if hasattr(obj, "keys") and hasattr(obj, "items") and "input_ids" in obj.keys():
        return dict(obj.items())
    if isinstance(obj, (list, tuple)):
        for item in obj:
            result = find_input_dict(item)
            if result is not None:
                return result
    return None


def align_mt_window(full_input_ids, mt_core_ids):
    occurrences = []
    window_size = len(mt_core_ids)

    for start_idx in range(len(full_input_ids) - window_size + 1):
        if full_input_ids[start_idx : start_idx + window_size] == mt_core_ids:
            occurrences.append(start_idx)

    return occurrences[-1] if occurrences else None


def main():
    args = parse_args()

    print("Loading ESA dataset...")
    loader = AnnotationLoader(refresh_cache=False)
    df = loader.get_view([args.dataset], only_overlap=False)
    error_col = f"{args.dataset}_error_spans"

    usable_df = df[df[error_col].apply(has_usable_spans)].copy()
    if args.limit > 0:
        usable_df = usable_df.head(args.limit)

    print(f"Usable samples after filtering unsupported/missing/undecided spans: {len(usable_df)}")

    print(f"Loading xCOMET model ({args.model})...")
    model_path = download_model(args.model)
    model = load_from_checkpoint(model_path)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    tokenizer = model.encoder.tokenizer

    all_vectors = []
    all_labels = []
    all_tokens = []
    all_offsets = []
    all_sample_ids = []
    label_counts = {
        "no_error": 0,
        "minor": 0,
        "major": 0,
        "critical": 0,
    }

    num_layers = None
    alignment_failures = 0

    for row_idx, row in tqdm(usable_df.iterrows(), total=len(usable_df), desc="Extracting"):
        source = row["source"]
        mt = row["hypothesis"]
        spans = filter_supported_spans(row[error_col])
        if not spans:
            continue

        enc = tokenizer(mt, return_offsets_mapping=True, add_special_tokens=True)
        labels = get_token_labels(spans, enc["offset_mapping"])
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])

        input_data = [{"src": source, "mt": mt, "ref": "", "score": 0.0}]
        prepared = model.prepare_sample(input_data)
        input_dict = find_input_dict(prepared)
        if input_dict is None:
            raise ValueError(f"Failed to locate input_ids for sample index {row_idx}.")

        if torch.cuda.is_available():
            input_dict = {
                key: value.to("cuda") if isinstance(value, torch.Tensor) else value
                for key, value in input_dict.items()
            }

        with torch.no_grad():
            outputs = model.encoder.model(
                input_ids=input_dict["input_ids"],
                attention_mask=input_dict["attention_mask"],
                output_hidden_states=True,
            )
            full_hidden_states = outputs.hidden_states
            if num_layers is None:
                num_layers = len(full_hidden_states)
                print(f"\n[INFO] Detected layers: {num_layers}")

        mt_core_ids = enc["input_ids"][1:-1]
        mt_core_offsets = enc["offset_mapping"][1:-1]
        mt_core_labels = labels[1:-1]
        mt_core_tokens = tokens[1:-1]

        full_ids = input_dict["input_ids"][0].detach().cpu().tolist()
        start_idx = align_mt_window(full_ids, mt_core_ids)
        if start_idx is None:
            alignment_failures += 1
            print(f"Warning: alignment failed for sample {row_idx}")
            continue

        stacked = torch.stack(full_hidden_states).squeeze(1)
        mt_all_layers = stacked[:, start_idx : start_idx + len(mt_core_ids), :].cpu().float()
        mt_all_layers = mt_all_layers.permute(1, 0, 2)

        sample_id = row.get("hypothesisID", f"row_{row_idx}")
        for token_idx in range(len(mt_core_ids)):
            label = mt_core_labels[token_idx]
            all_vectors.append(mt_all_layers[token_idx])
            all_labels.append(label)
            all_tokens.append(mt_core_tokens[token_idx])
            all_offsets.append(tuple(mt_core_offsets[token_idx]))
            all_sample_ids.append(sample_id)

            if label == NO_ERROR_LABEL:
                label_counts["no_error"] += 1
            elif label == SEVERITY_TO_LABEL["minor"]:
                label_counts["minor"] += 1
            elif label == SEVERITY_TO_LABEL["major"]:
                label_counts["major"] += 1
            elif label == SEVERITY_TO_LABEL["critical"]:
                label_counts["critical"] += 1

    if not all_vectors:
        raise RuntimeError("No token representations were extracted. Check dataset filtering settings.")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, args.output_name)

    print(f"\nStacking {len(all_vectors)} tokens...")
    x_final = torch.stack(all_vectors)
    y_final = torch.tensor(all_labels, dtype=torch.long)

    print(f"Saving to {output_file}...")
    torch.save(
        {
            "X": x_final,
            "y": y_final,
            "tokens": all_tokens,
            "offsets": all_offsets,
            "sample_ids": all_sample_ids,
            "label_map": {
                0: "no_error",
                1: "minor",
                2: "major",
                3: "critical",
            },
            "source_dataset": args.dataset,
            "model_name": args.model,
            "num_layers": num_layers,
            "label_counts": label_counts,
            "ignored_severities": sorted(IGNORED_SEVERITIES),
            "alignment_failures": alignment_failures,
        },
        output_file,
    )

    counts_df = pd.DataFrame(
        [
            {"label_id": 0, "label_name": "no_error", "count": label_counts["no_error"]},
            {"label_id": 1, "label_name": "minor", "count": label_counts["minor"]},
            {"label_id": 2, "label_name": "major", "count": label_counts["major"]},
            {"label_id": 3, "label_name": "critical", "count": label_counts["critical"]},
        ]
    )
    counts_file = os.path.join(output_dir, args.output_name.replace(".pt", "_label_counts.csv"))
    counts_df.to_csv(counts_file, index=False)

    print("\n" + "=" * 60)
    print("MULTILABEL EXTRACTION COMPLETE")
    print(f"Tensor shape: {x_final.shape}")
    print(f"Saved PT: {output_file}")
    print(f"Saved label counts: {counts_file}")
    print(f"Alignment failures: {alignment_failures}")
    print(f"Label counts: {label_counts}")
    print("=" * 60)


if __name__ == "__main__":
    main()
