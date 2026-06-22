import argparse
import os
import re

import h5py
import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_project_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def mat_string(value):
    return str(np.asarray(value).ravel()[0])


def parse_subject_ids(files):
    subject_ids = []
    pattern = re.compile(r"P(\d+)")

    for item in files.ravel():
        filename = mat_string(item)
        match = pattern.search(filename)
        if not match:
            raise ValueError(f"Cannot parse subject id from filename: {filename}")
        subject_ids.append(int(match.group(1)))

    return np.asarray(subject_ids, dtype=np.int64)


def load_uog20_samples(
    input_dir,
    spectrogram_file="Spectrograms.mat",
    label_file="Label.mat",
    spectrogram_key="Doppler2_MM",
    label_key="UpdatedLabel",
    image_height=240,
    num_classes=6,
):
    spectrogram_path = os.path.join(input_dir, spectrogram_file)
    label_path = os.path.join(input_dir, label_file)

    spectrogram_mat = loadmat(spectrogram_path)
    label_mat = loadmat(label_path)

    if spectrogram_key not in spectrogram_mat:
        raise KeyError(f"{spectrogram_key!r} not found in {spectrogram_path}")
    if label_key not in label_mat:
        raise KeyError(f"{label_key!r} not found in {label_path}")
    if "files" not in spectrogram_mat:
        raise KeyError(f"'files' not found in {spectrogram_path}")

    specs = np.asarray(spectrogram_mat[spectrogram_key], dtype=np.float32)
    labels = np.asarray(label_mat[label_key], dtype=np.int64).reshape(-1)
    subject_ids = parse_subject_ids(spectrogram_mat["files"])

    print("Raw specs shape:", specs.shape)
    print("Raw labels shape:", label_mat[label_key].shape)
    print("Raw files shape:", spectrogram_mat["files"].shape)

    if specs.ndim != 3:
        raise ValueError(
            "This script expects prepared UoG samples with shape "
            f"(num_samples, time, freq), but got {specs.shape}."
        )

    if specs.shape[0] != labels.shape[0] or specs.shape[0] != subject_ids.shape[0]:
        raise ValueError(
            "Sample count mismatch: "
            f"specs={specs.shape[0]}, labels={labels.shape[0]}, "
            f"files={subject_ids.shape[0]}"
        )

    # Current UoG .mat stores each sample as (time, freq). Models in this
    # project use (freq, time), so transpose once at load time.
    X = np.transpose(specs, (0, 2, 1))
    X = X[:, :image_height, :]

    y = labels - 1
    if y.min() < 0 or y.max() >= num_classes:
        raise ValueError(
            f"Labels should be in 1..{num_classes} before conversion, "
            f"but got converted range {y.min()}..{y.max()}."
        )

    X = np.asarray(
        [(spec - spec.mean()) / (spec.std() + 1e-8) for spec in X],
        dtype=np.float32,
    )

    return X, y, subject_ids


def split_id_ood_subjects(subject_ids, test_ood_subjects=2):
    unique_subjects = np.sort(np.unique(subject_ids))
    if test_ood_subjects <= 0:
        raise ValueError(f"test_ood_subjects must be positive, got {test_ood_subjects}.")
    if test_ood_subjects >= len(unique_subjects):
        raise ValueError(
            f"test_ood_subjects={test_ood_subjects} leaves no ID subjects "
            f"out of {len(unique_subjects)} total subjects."
        )

    return unique_subjects[:-test_ood_subjects], unique_subjects[-test_ood_subjects:]


def split_id_indices_by_subject(
    subject_ids,
    id_subjects,
    train_ratio=0.7,
    val_ratio=0.1,
    test_id_ratio=0.2,
):
    ratios = np.asarray([train_ratio, val_ratio, test_id_ratio], dtype=np.float64)
    if np.any(ratios <= 0):
        raise ValueError(f"Split ratios must be positive, got {ratios}.")
    ratios = ratios / ratios.sum()

    train_indices = []
    val_indices = []
    test_id_indices = []

    for subject_id in id_subjects:
        indices = np.where(subject_ids == subject_id)[0]
        if len(indices) < 3:
            raise ValueError(
                f"Subject {subject_id} has only {len(indices)} samples, "
                "cannot split into train/val/test_id."
            )

        train_count = int(np.floor(len(indices) * ratios[0]))
        val_count = int(np.floor(len(indices) * ratios[1]))
        test_count = len(indices) - train_count - val_count

        if train_count == 0:
            train_count = 1
            test_count -= 1
        if val_count == 0:
            val_count = 1
            test_count -= 1
        if test_count == 0:
            test_count = 1
            train_count -= 1

        train_end = train_count
        val_end = train_end + val_count

        train_indices.append(indices[:train_end])
        val_indices.append(indices[train_end:val_end])
        test_id_indices.append(indices[val_end:])

    return {
        "train": np.concatenate(train_indices),
        "val": np.concatenate(val_indices),
        "test_id": np.concatenate(test_id_indices),
    }


def sliding_window_samples(X, y, frame_length=300, stride=100, num_classes=6):
    """
    X shape: (N, freq, time)
    y shape: (N,)

    Each UoG sample has one action label, so every window inherits that
    sample-level class label as one-hot. No soft-labeling is needed here.
    """
    num_samples, freq_bins, time_steps = X.shape

    if time_steps < frame_length:
        raise ValueError(
            f"time_steps={time_steps} is smaller than frame_length={frame_length}."
        )

    starts = list(range(0, time_steps - frame_length + 1, stride))
    total_windows = num_samples * len(starts)

    X_windows = np.empty((total_windows, freq_bins, frame_length), dtype=np.float32)
    y_windows = np.empty((total_windows, num_classes), dtype=np.float32)

    index = 0
    for sample_idx in range(num_samples):
        label = y[sample_idx]
        label_one_hot = np.zeros(num_classes, dtype=np.float32)
        label_one_hot[label] = 1.0

        for start in starts:
            end = start + frame_length
            X_windows[index] = X[sample_idx, :, start:end]
            y_windows[index] = label_one_hot
            index += 1

    return X_windows, y_windows


def save_split_group(h5_file, group_name, X_windows, y_windows, subject_ids):
    group = h5_file.create_group(group_name)
    group.create_dataset("spectrograms", data=X_windows, compression="gzip")
    group.create_dataset("labels", data=y_windows, compression="gzip")
    group.create_dataset("subject_ids", data=subject_ids, compression="gzip")


def save_h5(output_h5, splits):
    os.makedirs(os.path.dirname(output_h5) or ".", exist_ok=True)

    with h5py.File(output_h5, "w") as f:
        for split_name in ("train", "val", "test_id", "test_ood"):
            split = splits[split_name]
            save_split_group(
                f,
                split_name,
                split["spectrograms"],
                split["labels"],
                split["subject_ids"],
            )


def prepare_uog20_subject_ood_h5(
    input_dir,
    output_h5,
    image_height=240,
    frame_length=300,
    stride=100,
    num_classes=6,
    test_ood_subjects=2,
    train_ratio=0.7,
    val_ratio=0.1,
    test_id_ratio=0.2,
):
    X, y, subject_ids = load_uog20_samples(
        input_dir=input_dir,
        image_height=image_height,
        num_classes=num_classes,
    )

    print("\nAfter loading:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("unique subjects:", len(np.unique(subject_ids)))

    id_subjects, test_ood_subjects_array = split_id_ood_subjects(
        subject_ids,
        test_ood_subjects=test_ood_subjects,
    )

    id_indices = split_id_indices_by_subject(
        subject_ids,
        id_subjects,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_id_ratio=test_id_ratio,
    )
    test_ood_indices = np.where(np.isin(subject_ids, test_ood_subjects_array))[0]

    split_indices = {
        "train": id_indices["train"],
        "val": id_indices["val"],
        "test_id": id_indices["test_id"],
        "test_ood": test_ood_indices,
    }

    prepared_splits = {}

    print("\nSubject split:")
    for split_name in ("train", "val", "test_id", "test_ood"):
        indices = split_indices[split_name]
        X_split = X[indices]
        y_split = y[indices]
        subject_ids_split = subject_ids[indices]
        split_subjects = np.sort(np.unique(subject_ids_split))

        X_windows, y_windows = sliding_window_samples(
            X_split,
            y_split,
            frame_length=frame_length,
            stride=stride,
            num_classes=num_classes,
        )

        windows_per_sample = X_windows.shape[0] // X_split.shape[0]
        window_subject_ids = np.repeat(subject_ids_split, windows_per_sample)

        prepared_splits[split_name] = {
            "spectrograms": X_windows,
            "labels": y_windows,
            "subject_ids": window_subject_ids,
        }

        print(
            f"{split_name}: subjects {split_subjects[0]}..{split_subjects[-1]} "
            f"({len(split_subjects)}), samples={X_split.shape[0]}, "
            f"windows={X_windows.shape[0]}"
        )

    print("\nAfter sliding window:")
    for split_name in ("train", "val", "test_id", "test_ood"):
        split = prepared_splits[split_name]
        print(f"{split_name}/spectrograms:", split["spectrograms"].shape)
        print(f"{split_name}/labels:", split["labels"].shape)

    print("\nExample one-hot train label:")
    print(prepared_splits["train"]["labels"][0])
    print("sum:", prepared_splits["train"]["labels"][0].sum())

    save_h5(output_h5, prepared_splits)
    print("\nSaved to:", output_h5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", default="datasets/Uog20/Dataset_848")
    parser.add_argument(
        "--output_h5",
        default="datasets/uog20_subject_ood.h5",
    )

    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--frame_length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=6)

    parser.add_argument("--test_ood_subjects", type=int, default=12)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_id_ratio", type=float, default=0.2)

    args = parser.parse_args()

    prepare_uog20_subject_ood_h5(
        input_dir=resolve_project_path(args.input_dir),
        output_h5=resolve_project_path(args.output_h5),
        image_height=args.image_height,
        frame_length=args.frame_length,
        stride=args.stride,
        num_classes=args.num_classes,
        test_ood_subjects=args.test_ood_subjects,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_id_ratio=args.test_id_ratio,
    )



