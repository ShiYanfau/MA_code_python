import argparse
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.fusion_dowm_patch import Fusion_Patch, Fusion_Patch_2
from model.mamba import MambaEncoder
from model.JEM import JEMWithLSTMDynamics
from model.LSTM_predictor import dynamics_loss
from utils.eval_plots import save_all_evaluation_plots


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "y")


def labels_to_class_index(y):
    if y.ndim == 2:
        return y.argmax(dim=1).long()
    return y.long()


class H5SampleDataset(Dataset):
    def __init__(self, h5_path, split, normalize_uint8=False):
        self.h5_path = h5_path
        self.split = split
        self.normalize_uint8 = normalize_uint8

        with h5py.File(self.h5_path, "r") as f:
            group = f[self.split]
            if "sample_ids" in group:
                self.group_id_key = "sample_ids"
            elif "source_ids" in group:
                self.group_id_key = "source_ids"
            elif "subject_ids" in group:
                self.group_id_key = "subject_ids"
            else:
                self.group_id_key = None

            n = group["spectrograms"].shape[0]
            if self.group_id_key is None:
                group_ids = np.arange(n)
            else:
                group_ids = group[self.group_id_key][:]

        self.group_ids = group_ids
        self.unique_group_ids = np.unique(group_ids)
        self.group_to_indices = {
            group_id: np.where(group_ids == group_id)[0]
            for group_id in self.unique_group_ids
        }

        print(
            f"{split}: using group key = {self.group_id_key}, "
            f"windows = {len(group_ids)}, groups = {len(self.unique_group_ids)}"
        )

    def __len__(self):
        return len(self.unique_group_ids)

    def __getitem__(self, idx):
        group_id = self.unique_group_ids[idx]
        indices = self.group_to_indices[group_id]

        with h5py.File(self.h5_path, "r") as f:
            x = f[self.split]["spectrograms"][indices]
            y = f[self.split]["labels"][indices]

        if x.dtype == np.uint8 and self.normalize_uint8:
            x = x.astype(np.float32) / 255.0
        else:
            x = x.astype(np.float32)

        x = torch.tensor(x, dtype=torch.float32)

        y = torch.tensor(y)
        if y.ndim == 2:
            y_idx = y.argmax(dim=1).long()
        else:
            y_idx = y.long()

        values, counts = torch.unique(y_idx, return_counts=True)
        sample_label = values[counts.argmax()].long()

        return x, sample_label, torch.tensor(group_id, dtype=torch.long)


def sample_collate_fn(batch):
    xs, ys, group_ids = zip(*batch)

    batch_size = len(xs)
    max_windows = max(x.shape[0] for x in xs)
    sample_shape = xs[0].shape[1:]

    x_padded = torch.zeros((batch_size, max_windows, *sample_shape), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_windows), dtype=torch.bool)

    for i, x in enumerate(xs):
        n = x.shape[0]
        x_padded[i, :n] = x
        mask[i, :n] = True

    y = torch.stack(ys).long()
    group_ids = torch.stack(group_ids).long()

    return x_padded, y, mask, group_ids


def build_model(dataset_name, num_classes, embed_dim, mamba_layers, dropout):
    if dataset_name == "diat":
        patch_encoder = Fusion_Patch(embed_dim=embed_dim)
    elif dataset_name == "uog":
        patch_encoder = Fusion_Patch_2(
            freq_bins=43,
            embed_dim=embed_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    encoder = nn.Sequential(
        patch_encoder,
        MambaEncoder(
            d_model=embed_dim,
            num_layers=mamba_layers,
            d_state=16,
            d_conv=4,
            expand=2,
            dropout=dropout,
        ),
    )

    return JEMWithLSTMDynamics(
        encoder=encoder,
        latent_dim=embed_dim,
        num_classes=num_classes,
        dynamics_hidden_dim=256,
        dynamics_num_layers=2,
        dropout=dropout,
    )


def forward_sample_batch(model, x, mask):
    b, w = x.shape[:2]
    x_flat = x[mask]

    logits_flat, z_flat, z_pred_flat = model(x_flat)

    counts = mask.sum(dim=1).tolist()
    logits_split = torch.split(logits_flat, counts, dim=0)

    sample_logits = torch.stack([
        item.mean(dim=0)
        for item in logits_split
    ], dim=0)

    energy_flat = -torch.logsumexp(logits_flat, dim=1)
    dyn_flat = ((z_pred_flat - z_flat[:, 1:, :]) ** 2).mean(dim=(1, 2))

    energy_split = torch.split(energy_flat, counts, dim=0)
    dyn_split = torch.split(dyn_flat, counts, dim=0)

    sample_energy = torch.stack([item.mean() for item in energy_split])
    sample_dyn = torch.stack([item.mean() for item in dyn_split])

    return sample_logits, sample_energy, sample_dyn, z_flat, z_pred_flat


def train_one_epoch(model, loader, optimizer, device, lambda_dyn, epoch=None, log_interval=10):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_dyn_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(loader)

    for batch_idx, (x, y, mask, _) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        sample_logits, _, _, z_flat, z_pred_flat = forward_sample_batch(model, x, mask)

        cls_loss = ce_loss_fn(sample_logits, y)
        dyn_loss = dynamics_loss(z_pred_flat, z_flat)
        loss = cls_loss + lambda_dyn * dyn_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        pred = sample_logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_dyn_loss += dyn_loss.item() * batch_size
        correct += (pred == y).sum().item()
        total += batch_size

        if batch_idx % log_interval == 0 or batch_idx == 1 or batch_idx == num_batches:
            epoch_text = f"Epoch {epoch} " if epoch is not None else ""
            print(
                f"{epoch_text}Train batch [{batch_idx:03d}/{num_batches:03d}] "
                f"loss={total_loss / total:.4f} "
                f"cls_loss={total_cls_loss / total:.4f} "
                f"dyn_loss={total_dyn_loss / total:.4f} "
                f"sample_acc={correct / total:.4f}",
                flush=True,
            )

    return {
        "loss": total_loss / total,
        "cls_loss": total_cls_loss / total,
        "dyn_loss": total_dyn_loss / total,
        "acc": correct / total,
    }


@torch.no_grad()
def evaluate_id(model, loader, device, lambda_dyn):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_dyn_loss = 0.0
    correct = 0
    total = 0

    all_energy = []
    all_dyn = []
    all_true = []
    all_pred = []

    for x, y, mask, _ in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        sample_logits, sample_energy, sample_dyn, z_flat, z_pred_flat = forward_sample_batch(model, x, mask)

        cls_loss = ce_loss_fn(sample_logits, y)
        dyn_loss = dynamics_loss(z_pred_flat, z_flat)
        loss = cls_loss + lambda_dyn * dyn_loss

        pred = sample_logits.argmax(dim=1)
        batch_size = y.size(0)

        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_dyn_loss += dyn_loss.item() * batch_size
        correct += (pred == y).sum().item()
        total += batch_size

        all_energy.append(sample_energy.cpu())
        all_dyn.append(sample_dyn.cpu())
        all_true.append(y.cpu())
        all_pred.append(pred.cpu())

    return {
        "loss": total_loss / total,
        "cls_loss": total_cls_loss / total,
        "dyn_loss": total_dyn_loss / total,
        "acc": correct / total,
        "energy": torch.cat(all_energy).numpy(),
        "dyn_error": torch.cat(all_dyn).numpy(),
        "y_true": torch.cat(all_true).numpy(),
        "y_pred": torch.cat(all_pred).numpy(),
    }


@torch.no_grad()
def evaluate_ood(model, loader, device):
    model.eval()

    all_energy = []
    all_dyn = []

    for x, _, mask, _ in loader:
        x = x.to(device)
        mask = mask.to(device)

        _, sample_energy, sample_dyn, _, _ = forward_sample_batch(model, x, mask)

        all_energy.append(sample_energy.cpu())
        all_dyn.append(sample_dyn.cpu())

    return {
        "energy": torch.cat(all_energy).numpy(),
        "dyn_error": torch.cat(all_dyn).numpy(),
    }


def normalize_by_id(id_scores, scores):
    return (scores - id_scores.mean()) / (id_scores.std() + 1e-8)


def try_compute_auroc(id_scores, ood_scores):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None

    y_true = np.concatenate([
        np.zeros_like(id_scores),
        np.ones_like(ood_scores),
    ])
    y_score = np.concatenate([id_scores, ood_scores])

    return roc_auc_score(y_true, y_score)


def save_training_curves(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Sample-level Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "sl_loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_sample_acc")
    plt.plot(epochs, history["val_acc"], label="val_sample_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Sample-level Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "sl_acc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["diat", "uog"], default="uog")
    parser.add_argument("--h5_path", type=str, default="datasets/uog20_subject_ood.h5")
    parser.add_argument("--num_classes", type=int, default=5)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_dyn", type=float, default=0.1)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="checkpoints/best_model_uog_sample_level.pt")
    parser.add_argument("--plot_dir", type=str, default="results/uog_sl")
    parser.add_argument("--normalize_uint8", action="store_true")
    parser.add_argument("--test_only", type=str_to_bool, default=False)

    args = parser.parse_args()

    if args.dataset == "uog" and args.num_classes == 5:
        args.num_classes = 6

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Dataset:", args.dataset)
    print("H5 path:", args.h5_path)
    print("Num classes:", args.num_classes)
    print("Mode: sample-level train/test")

    train_dataset = H5SampleDataset(args.h5_path, "train", normalize_uint8=args.normalize_uint8)
    val_dataset = H5SampleDataset(args.h5_path, "val", normalize_uint8=args.normalize_uint8)
    test_id_dataset = H5SampleDataset(args.h5_path, "test_id", normalize_uint8=args.normalize_uint8)
    test_ood_dataset = H5SampleDataset(args.h5_path, "test_ood", normalize_uint8=args.normalize_uint8)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=sample_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=sample_collate_fn,
    )
    test_id_loader = DataLoader(
        test_id_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=sample_collate_fn,
    )
    test_ood_loader = DataLoader(
        test_ood_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=sample_collate_fn,
    )

    model = build_model(
        dataset_name=args.dataset,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        mamba_layers=args.mamba_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    if not args.test_only:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                lambda_dyn=args.lambda_dyn,
                epoch=epoch,
            )

            val_metrics = evaluate_id(
                model=model,
                loader=val_loader,
                device=device,
                lambda_dyn=args.lambda_dyn,
            )

            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_sample_acc={train_metrics['acc']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_sample_acc={val_metrics['acc']:.4f}"
            )

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["acc"])

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_val_sample_acc": best_val_acc,
                        "args": vars(args),
                    },
                    args.save_path,
                )
                print(f"Saved best sample-level model to {args.save_path}")

            save_training_curves(history, args.plot_dir)

    print("\nLoading best model...")
    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_id_metrics = evaluate_id(
        model=model,
        loader=test_id_loader,
        device=device,
        lambda_dyn=args.lambda_dyn,
    )

    test_ood_metrics = evaluate_ood(
        model=model,
        loader=test_ood_loader,
        device=device,
    )

    id_energy = test_id_metrics["energy"]
    ood_energy = test_ood_metrics["energy"]

    id_dyn = test_id_metrics["dyn_error"]
    ood_dyn = test_ood_metrics["dyn_error"]

    id_energy_norm = normalize_by_id(id_energy, id_energy)
    ood_energy_norm = normalize_by_id(id_energy, ood_energy)

    id_dyn_norm = normalize_by_id(id_dyn, id_dyn)
    ood_dyn_norm = normalize_by_id(id_dyn, ood_dyn)

    id_fusion = 0.5 * id_energy_norm + 0.5 * id_dyn_norm
    ood_fusion = 0.5 * ood_energy_norm + 0.5 * ood_dyn_norm

    energy_auroc = try_compute_auroc(id_energy, ood_energy)
    dyn_auroc = try_compute_auroc(id_dyn, ood_dyn)
    fusion_auroc = try_compute_auroc(id_fusion, ood_fusion)

    print("\nSample-level Test ID:")
    print(f"loss: {test_id_metrics['loss']:.4f}")
    print(f"sample-level acc: {test_id_metrics['acc']:.4f}")

    print("\nSample-level OOD score statistics:")
    print(f"ID energy mean:  {id_energy.mean():.4f}")
    print(f"OOD energy mean: {ood_energy.mean():.4f}")
    print(f"ID dyn mean:     {id_dyn.mean():.4f}")
    print(f"OOD dyn mean:    {ood_dyn.mean():.4f}")

    if energy_auroc is not None:
        print("\nSample-level OOD AUROC:")
        print(f"energy AUROC:   {energy_auroc:.4f}")
        print(f"dynamics AUROC: {dyn_auroc:.4f}")
        print(f"fusion AUROC:   {fusion_auroc:.4f}")
    else:
        print("\nsklearn not installed, skipped AUROC.")

    class_names = [f"Class {i}" for i in range(args.num_classes)]
    save_all_evaluation_plots(
        y_true=test_id_metrics["y_true"],
        y_pred=test_id_metrics["y_pred"],
        id_energy=id_energy,
        ood_energy=ood_energy,
        id_dyn=id_dyn,
        ood_dyn=ood_dyn,
        id_fusion=id_fusion,
        ood_fusion=ood_fusion,
        save_dir=args.plot_dir,
        class_names=class_names,
    )

    print(f"\nSaved sample-level plots to: {args.plot_dir}")


if __name__ == "__main__":
    main()


# 这是uogsl的测试和训练程序

































