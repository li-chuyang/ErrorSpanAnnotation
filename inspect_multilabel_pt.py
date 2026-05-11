import argparse
from collections import Counter, defaultdict

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect and sanity-check a multilabel token-level PT extraction file."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="extract_multilabel_output/experiment_esa1_all_layers_multilabel.pt",
        help="Path to the extracted multilabel PT file.",
    )
    parser.add_argument(
        "--examples-per-label",
        type=int,
        default=5,
        help="How many example tokens to print for each label.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading PT with mmap: {args.input_file}")
    data = torch.load(args.input_file, mmap=True)

    x = data["X"]
    y = data["y"]
    tokens = data["tokens"]
    offsets = data["offsets"]
    sample_ids = data["sample_ids"]
    label_map = data.get("label_map", {})

    print("\n" + "=" * 70)
    print("BASIC SHAPE CHECK")
    print("=" * 70)
    print(f"X shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"num_layers metadata: {data.get('num_layers')}")
    print(f"model_name: {data.get('model_name')}")
    print(f"source_dataset: {data.get('source_dataset')}")
    print(f"alignment_failures: {data.get('alignment_failures')}")
    print(f"ignored_severities: {data.get('ignored_severities')}")

    assert x.ndim == 3, f"Expected X to be 3D, got {x.ndim}D"
    assert y.ndim == 1, f"Expected y to be 1D, got {y.ndim}D"
    assert x.shape[0] == y.shape[0], "Token count mismatch between X and y"
    assert x.shape[0] == len(tokens), "Token count mismatch between X and tokens"
    assert x.shape[0] == len(offsets), "Token count mismatch between X and offsets"
    assert x.shape[0] == len(sample_ids), "Token count mismatch between X and sample_ids"

    print("\nAll basic shape checks passed.")

    y_list = y.tolist()
    label_counter = Counter(y_list)
    unique_labels = sorted(label_counter.keys())

    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)
    for label_id in unique_labels:
        label_name = label_map.get(label_id, f"label_{label_id}")
        print(f"{label_id:>2} | {label_name:<10} | {label_counter[label_id]:>6}")

    print("\nSaved label_counts metadata:")
    print(data.get("label_counts"))

    per_label_examples = defaultdict(list)
    for idx, label_id in enumerate(y_list):
        if len(per_label_examples[label_id]) >= args.examples_per_label:
            continue

        per_label_examples[label_id].append(
            {
                "index": idx,
                "token": tokens[idx],
                "offset": offsets[idx],
                "sample_id": sample_ids[idx],
                "vector_shape": tuple(x[idx].shape),
            }
        )

        if all(len(per_label_examples[label]) >= args.examples_per_label for label in unique_labels):
            break

    print("\n" + "=" * 70)
    print("EXAMPLE TOKENS BY LABEL")
    print("=" * 70)
    for label_id in unique_labels:
        label_name = label_map.get(label_id, f"label_{label_id}")
        print(f"\nLabel {label_id} ({label_name})")
        for example in per_label_examples[label_id]:
            print(
                f"  idx={example['index']:<6} "
                f"token={example['token']:<20} "
                f"offset={example['offset']!s:<14} "
                f"sample={example['sample_id']} "
                f"vec={example['vector_shape']}"
            )

    print("\n" + "=" * 70)
    print("SAMPLE TOKEN WINDOWS")
    print("=" * 70)
    shown_samples = set()
    for idx, sample_id in enumerate(sample_ids):
        if sample_id in shown_samples:
            continue

        start = idx
        while start > 0 and sample_ids[start - 1] == sample_id:
            start -= 1

        end = idx
        while end + 1 < len(sample_ids) and sample_ids[end + 1] == sample_id:
            end += 1

        shown_samples.add(sample_id)
        print(f"\nSample: {sample_id}")
        preview_end = min(start + 20, end + 1)
        for pos in range(start, preview_end):
            label_id = y_list[pos]
            label_name = label_map.get(label_id, f"label_{label_id}")
            print(
                f"  {pos:>6} | token={tokens[pos]:<20} "
                f"label={label_id}({label_name}) offset={offsets[pos]}"
            )

        if len(shown_samples) >= 3:
            break

    print("\n" + "=" * 70)
    print("PT INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
