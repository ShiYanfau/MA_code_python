import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.fusion_dowm_patch import Fusion_Patch, Fusion_Patch_2
from model.mamba import MambaEncoder
from model.JEM import JEMWithLSTMDynamics
from model.LSTM_predictor import dynamics_loss
from utils.eval_plots import save_all_evaluation_plots


class H5SplitDataset(Dataset):
    def __init__(self, h5_path, split, normalize_uint8=False):
        self.h5_path = h5_path
        self.split = split
        self.normalize_uint8 = normalize_uint8

        with h5py.File(self.h5_path, "r") as f:
            self.length = f[self.split]["spectrograms"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            x = f[self.split]["spectrograms"][idx]
            y = f[self.split]["labels"][idx]

        if x.dtype == np.uint8 and self.normalize_uint8:
            x = x.astype(np.float32) / 255.0
        else:
            x = x.astype(np.float32)

        x = torch.tensor(x, dtype=torch.float32)

        if np.ndim(y) == 0:
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = torch.tensor(y, dtype=torch.float32)

        return x, y


def labels_to_class_index(y):
    if y.ndim == 2:
        return y.argmax(dim=1).long()
    return y.long()


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

    model = JEMWithLSTMDynamics(
        encoder=encoder,
        latent_dim=embed_dim,
        num_classes=num_classes,
        dynamics_hidden_dim=256,
        dynamics_num_layers=2,
        dropout=dropout,
    )

    return model


def train_one_epoch(model, loader, optimizer, device, lambda_dyn, epoch=None, log_interval=10):
    model.train()

    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_dyn_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(loader)

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        target = labels_to_class_index(y)

        # logits, z, z_pred = model(x)
        #
        # cls_loss = ce_loss_fn(logits, target)
        # dyn_loss = dynamics_loss(z_pred, z)
        # loss = cls_loss + lambda_dyn * dyn_loss

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN/Inf found in input x")
            print("x min:", torch.nan_to_num(x).min().item())
            print("x max:", torch.nan_to_num(x).max().item())
            raise ValueError("Bad input x")

        with torch.no_grad():
            patch_out = model.encoder[0](x)

            if torch.isnan(patch_out).any() or torch.isinf(patch_out).any():
                print("NaN/Inf found after Fusion_Patch")
                print("patch_out shape:", patch_out.shape)
                print("patch_out min:", torch.nan_to_num(patch_out).min().item())
                print("patch_out max:", torch.nan_to_num(patch_out).max().item())
                raise ValueError("Bad Fusion_Patch output")

            mamba_out = model.encoder[1](patch_out)

            if torch.isnan(mamba_out).any() or torch.isinf(mamba_out).any():
                print("NaN/Inf found after MambaEncoder")
                print("mamba_out shape:", mamba_out.shape)
                print("mamba_out min:", torch.nan_to_num(mamba_out).min().item())
                print("mamba_out max:", torch.nan_to_num(mamba_out).max().item())
                raise ValueError("Bad MambaEncoder output")

        logits, z, z_pred = model(x)

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN/Inf found in input x")
            print("x min:", x.nanmin().item(), "x max:", x.nanmax().item())
            raise ValueError("Bad input x")

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("NaN/Inf found in logits")
            raise ValueError("Bad logits")

        if torch.isnan(z).any() or torch.isinf(z).any():
            print("NaN/Inf found in z")
            raise ValueError("Bad latent z")

        if torch.isnan(z_pred).any() or torch.isinf(z_pred).any():
            print("NaN/Inf found in z_pred")
            raise ValueError("Bad z_pred")

        cls_loss = ce_loss_fn(logits, target)
        dyn_loss = dynamics_loss(z_pred, z)

        if torch.isnan(cls_loss) or torch.isinf(cls_loss):
            print("NaN/Inf found in cls_loss")
            raise ValueError("Bad cls_loss")

        if torch.isnan(dyn_loss) or torch.isinf(dyn_loss):
            print("NaN/Inf found in dyn_loss")
            print("z min:", z.nanmin().item(), "z max:", z.nanmax().item())
            print("z_pred min:", z_pred.nanmin().item(), "z_pred max:", z_pred.nanmax().item())
            raise ValueError("Bad dyn_loss")

        loss = cls_loss + lambda_dyn * dyn_loss
        ##

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)

        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_dyn_loss += dyn_loss.item() * batch_size

        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += batch_size

        if batch_idx % log_interval == 0 or batch_idx == 1 or batch_idx == num_batches:
            epoch_text = f"Epoch {epoch} " if epoch is not None else ""
            print(
                f"{epoch_text}Train batch [{batch_idx:03d}/{num_batches:03d}] "
                f"loss={total_loss / total:.4f} "
                f"cls_loss={total_cls_loss / total:.4f} "
                f"dyn_loss={total_dyn_loss / total:.4f} "
                f"acc={correct / total:.4f}",
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
    all_dyn_error = []
    all_targets = []
    all_preds = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        target = labels_to_class_index(y)

        logits, z, z_pred = model(x)

        cls_loss = ce_loss_fn(logits, target)
        dyn_loss = dynamics_loss(z_pred, z)
        loss = cls_loss + lambda_dyn * dyn_loss

        energy = -torch.logsumexp(logits, dim=1)
        dyn_error = ((z_pred - z[:, 1:, :]) ** 2).mean(dim=(1, 2))

        batch_size = x.size(0)

        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_dyn_loss += dyn_loss.item() * batch_size

        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += batch_size

        all_energy.append(energy.cpu())
        all_dyn_error.append(dyn_error.cpu())
        all_targets.append(target.cpu())
        all_preds.append(pred.cpu())

    return {
        "loss": total_loss / total,
        "cls_loss": total_cls_loss / total,
        "dyn_loss": total_dyn_loss / total,
        "acc": correct / total,
        "energy": torch.cat(all_energy).numpy(),
        "dyn_error": torch.cat(all_dyn_error).numpy(),
        "y_true": torch.cat(all_targets).numpy(),
        "y_pred": torch.cat(all_preds).numpy(),
    }


@torch.no_grad()
def evaluate_ood(model, loader, device):
    model.eval()

    all_energy = []
    all_dyn_error = []

    for x, _ in loader:
        x = x.to(device)

        logits, z, z_pred = model(x)

        energy = -torch.logsumexp(logits, dim=1)
        dyn_error = ((z_pred - z[:, 1:, :]) ** 2).mean(dim=(1, 2))

        all_energy.append(energy.cpu())
        all_dyn_error.append(dyn_error.cpu())

    return {
        "energy": torch.cat(all_energy).numpy(),
        "dyn_error": torch.cat(all_dyn_error).numpy(),
    }


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

def normalize_by_id(id_scores, scores):
    """
    用 ID 测试集的均值和标准差归一化 OOD 分数。
    注意：不要用 OOD 的均值和标准差归一化，否则会信息泄露。
    """
    return (scores - id_scores.mean()) / (id_scores.std() + 1e-8)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",choices=["diat", "uog"],default="diat",) # 选择数据集，diat 或 uog
    parser.add_argument("--h5_path",type=str,default="datasets/diat_processed_ood_StonepeltingGrenadesthrowing.h5",) # H5 文件路径
    parser.add_argument("--num_classes", type=int, default=5) # 分类数量，diat 是 5 类，uog 是 6 类（但 uog 的标签文件里是 one-hot，所以默认也是 5）

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_dyn", type=float, default=0.1)

    parser.add_argument("--num_workers", type=int, default=14)
    parser.add_argument("--save_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--plot_dir", type=str, default="result/eval_plots")
    parser.add_argument("--normalize_uint8", action="store_true")
    parser.add_argument("--test_only", type=bool, default=False)

    args = parser.parse_args()

    if args.dataset == "uog" and args.num_classes == 5:
        args.num_classes = 6

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    print("Device:", device)
    print("Dataset:", args.dataset)
    print("H5 path:", args.h5_path)
    print("Num classes:", args.num_classes)

    train_dataset = H5SplitDataset(args.h5_path, "train", normalize_uint8=args.normalize_uint8) # 创建训练集、验证集、测试集的 Dataset 实例
    val_dataset = H5SplitDataset(args.h5_path, "val", normalize_uint8=args.normalize_uint8)
    test_id_dataset = H5SplitDataset(args.h5_path, "test_id", normalize_uint8=args.normalize_uint8)
    test_ood_dataset = H5SplitDataset(args.h5_path, "test_ood", normalize_uint8=args.normalize_uint8)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_id_loader = DataLoader(
        test_id_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_ood_loader = DataLoader(
        test_ood_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(    # 创建模型实例
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

    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_cls_loss": [],
        "train_dyn_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_cls_loss": [],
        "val_dyn_loss": [],
        "val_acc": [],
    }

    if not args.test_only:
        for epoch in range(1, args.epochs + 1):  # 训练循环，每个 epoch 包括训练和验证
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                lambda_dyn=args.lambda_dyn,
                epoch=epoch,
                log_interval=10,
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
                f"train_acc={train_metrics['acc']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['acc']:.4f} "
                f"val_dyn={val_metrics['dyn_loss']:.4f}"
            )

            history["train_loss"].append(train_metrics["loss"])
            history["train_cls_loss"].append(train_metrics["cls_loss"])
            history["train_dyn_loss"].append(train_metrics["dyn_loss"])
            history["train_acc"].append(train_metrics["acc"])

            history["val_loss"].append(val_metrics["loss"])
            history["val_cls_loss"].append(val_metrics["cls_loss"])
            history["val_dyn_loss"].append(val_metrics["dyn_loss"])
            history["val_acc"].append(val_metrics["acc"])

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_val_acc": best_val_acc,
                        "args": vars(args),
                    },
                    args.save_path,
                )

                print(f"Saved best model to {args.save_path}")

            os.makedirs("result", exist_ok=True)

            epochs = range(1, len(history["train_loss"]) + 1)

            plt.figure()
            plt.plot(epochs, history["train_loss"], label="train_loss")
            plt.plot(epochs, history["val_loss"], label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig("result/loss_curve.png", dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure()
            plt.plot(epochs, history["train_cls_loss"], label="train_cls_loss")
            plt.plot(epochs, history["val_cls_loss"], label="val_cls_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Classification Loss")
            plt.title("Classification Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig("result/cls_loss_curve.png", dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure()
            plt.plot(epochs, history["train_dyn_loss"], label="train_dyn_loss")
            plt.plot(epochs, history["val_dyn_loss"], label="val_dyn_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Dynamics Loss")
            plt.title("Dynamics Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig("result/dyn_loss_curve.png", dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure()
            plt.plot(epochs, history["train_acc"], label="train_acc")
            plt.plot(epochs, history["val_acc"], label="val_acc")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Training and Validation Accuracy")
            plt.legend()
            plt.grid(True)
            plt.savefig("result/acc_curve.png", dpi=300, bbox_inches="tight")
            plt.close()

            print("Saved training curves to result/")




    print("\nLoading best model...")
    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_id_metrics = evaluate_id(
        model=model,
        loader=test_id_loader,
        device=device,
        lambda_dyn=args.lambda_dyn,
    )  # 在测试集上评估模型，得到 ID

    test_ood_metrics = evaluate_ood(
        model=model,
        loader=test_ood_loader,
        device=device,
    )

    print("\nTest ID:")
    print(f"loss: {test_id_metrics['loss']:.4f}")
    print(f"acc:  {test_id_metrics['acc']:.4f}")

    id_energy = test_id_metrics["energy"]
    ood_energy = test_ood_metrics["energy"]

    id_dyn = test_id_metrics["dyn_error"]
    ood_dyn = test_ood_metrics["dyn_error"]

    print("\nOOD score statistics:")
    print(f"ID energy mean:  {id_energy.mean():.4f}")
    print(f"OOD energy mean: {ood_energy.mean():.4f}")
    print(f"ID dyn mean:     {id_dyn.mean():.4f}")
    print(f"OOD dyn mean:    {ood_dyn.mean():.4f}")

    energy_auroc = try_compute_auroc(id_energy, ood_energy)
    dyn_auroc = try_compute_auroc(id_dyn, ood_dyn)

    # ===== OOD score fusion =====
    # 先用 ID 分布归一化两个分数，避免 energy 和 dynamics error 数值尺度不同
    id_energy_norm = normalize_by_id(id_energy, id_energy)
    ood_energy_norm = normalize_by_id(id_energy, ood_energy)

    id_dyn_norm = normalize_by_id(id_dyn, id_dyn)
    ood_dyn_norm = normalize_by_id(id_dyn, ood_dyn)

    # 简单等权融合
    alpha = 0.5
    beta = 0.5

    id_fusion_score = alpha * id_energy_norm + beta * id_dyn_norm
    ood_fusion_score = alpha * ood_energy_norm + beta * ood_dyn_norm

    fusion_auroc = try_compute_auroc(id_fusion_score, ood_fusion_score)

    if energy_auroc is not None:
        print("\nOOD AUROC:")
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
        id_fusion=id_fusion_score,
        ood_fusion=ood_fusion_score,
        save_dir=args.plot_dir,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()





































