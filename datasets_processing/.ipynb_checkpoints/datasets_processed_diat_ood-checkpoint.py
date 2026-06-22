import os
import argparse
import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


DIAT_CLASSES = [
    "Army crawling",
    "Army jogging",
    "Army marching",
    "Boxing",
    "Jumping with holding a gun",
    "Stone pelting-Grenades throwing",
]


def make_safe_class_name(class_name: str) -> str:
    return "".join(ch for ch in class_name if ch.isalnum())


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    图片预处理流程：
    RGB 转换 -> resize 到 (H=128, W=256) -> HWC 转 CHW -> min-max 归一化

    返回 shape: (3, 128, 256), dtype: float32, 范围: [0, 1]
    其中 W=256 作为时间轴。
    """
    img = Image.open(image_path).convert("RGB")

    # PIL resize 的 size 参数是 (width, height)
    img = img.resize((256, 128), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32)  # HWC: (128, 256, 3)
    arr = np.transpose(arr, (2, 0, 1))       # CHW: (3, 128, 256)

    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    return arr.astype(np.float32)


def sliding_window_image(
    image: np.ndarray,
    window_size: int = 128,
    stride: int = 64,
) -> np.ndarray:
    """
    沿时间维 W 做滑窗。

    输入:
        image: shape (3, 128, 256)

    输出:
        windows: shape (num_windows, 3, 128, 128)

    当 W=256, window_size=128, stride=64 时：
        第 1 个窗口: [:, :, 0:128]
        第 2 个窗口: [:, :, 64:192]
        第 3 个窗口: [:, :, 128:256]
    """
    if image.ndim != 3:
        raise ValueError(f"期望输入图片为 CHW 三维数组，实际 shape={image.shape}")

    _, _, width = image.shape
    if width < window_size:
        raise ValueError(
            f"图片时间维长度不足，width={width}, window_size={window_size}"
        )

    windows = []
    for start in range(0, width - window_size + 1, stride):
        end = start + window_size
        windows.append(image[:, :, start:end])

    if len(windows) == 0:
        raise RuntimeError(
            f"滑窗后没有得到任何窗口，image_shape={image.shape}, "
            f"window_size={window_size}, stride={stride}"
        )

    return np.asarray(windows, dtype=np.float32)


def collect_class_samples(
    base_dir: str,
    class_name: str,
    class_idx: int,
    window_size: int,
    stride: int,
    start_source_id: int,
):
    """
    读取某个类别下的 figure*.jpg，并完成：

    原图读取
    -> RGB 转换
    -> resize 到 (3, 128, 256)
    -> min-max 归一化
    -> 滑窗分图

    每个滑窗图会得到：
        1) 分类标签 labels: class_idx
        2) source_ids: 表示它来自哪一张原始图片

    同一张原始图片滑出来的多个窗口拥有相同 source_id。
    """
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_dir):
        raise FileNotFoundError(f"类别目录不存在: {class_dir}")

    images, labels, source_ids = [], [], []
    current_source_id = start_source_id

    for fname in sorted(os.listdir(class_dir)):
        if fname.startswith("figure") and fname.lower().endswith(".jpg"):
            fpath = os.path.join(class_dir, fname)
            try:
                x = load_and_preprocess_image(fpath)
                windows = sliding_window_image(
                    x,
                    window_size=window_size,
                    stride=stride,
                )

                for win in windows:
                    images.append(win)
                    labels.append(class_idx)
                    source_ids.append(current_source_id)

                current_source_id += 1

            except Exception as e:
                print(f"[WARN] 跳过 {fpath}: {e}")

    if len(images) == 0:
        raise RuntimeError(f"类别 {class_name} 下没有可用样本。")

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    source_ids = np.asarray(source_ids, dtype=np.int64)

    return X, y, source_ids, current_source_id


def split_per_class(X: np.ndarray, y: np.ndarray, source_ids: np.ndarray):
    """
    先 80/20 分 trainval/test，再从 trainval 分 87.5/12.5 得到 train/val
    => 总体约为 70/10/20

    注意：
    这里 shuffle=False。
    因为同一原始样本滑窗得到的窗口是连续排列的，所以默认情况下
    同源窗口会尽量保持在同一个切分区域中。
    """
    X_trainval, X_test, y_trainval, y_test, sid_trainval, sid_test = train_test_split(
        X,
        y,
        source_ids,
        test_size=0.2,
        shuffle=False,
    )

    X_train, X_val, y_train, y_val, sid_train, sid_val = train_test_split(
        X_trainval,
        y_trainval,
        sid_trainval,
        test_size=0.125,
        shuffle=False,
    )

    return (
        X_train,
        y_train,
        sid_train,
        X_val,
        y_val,
        sid_val,
        X_test,
        y_test,
        sid_test,
    )


def one_hot_encode_labels(y_train, y_val, y_test, num_classes: int):
    encoder = OneHotEncoder(
        sparse_output=False,
        categories=[np.arange(num_classes)],
    )
    y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
    y_val_oh = encoder.transform(y_val.reshape(-1, 1)).astype(np.float32)
    y_test_oh = encoder.transform(y_test.reshape(-1, 1)).astype(np.float32)
    return y_train_oh, y_val_oh, y_test_oh


def save_h5_classification(
    output_h5: str,
    X_train,
    y_train,
    sid_train,
    X_val,
    y_val,
    sid_val,
    X_test,
    y_test,
    sid_test,
):
    os.makedirs(os.path.dirname(output_h5) or ".", exist_ok=True)

    with h5py.File(output_h5, "w") as f:
        g = f.create_group("train")
        g.create_dataset("spectrograms", data=X_train, dtype="float32")
        g.create_dataset("labels", data=y_train, dtype="float32")
        g.create_dataset("source_ids", data=sid_train, dtype="int64")

        g = f.create_group("val")
        g.create_dataset("spectrograms", data=X_val, dtype="float32")
        g.create_dataset("labels", data=y_val, dtype="float32")
        g.create_dataset("source_ids", data=sid_val, dtype="int64")

        g = f.create_group("test")
        g.create_dataset("spectrograms", data=X_test, dtype="float32")
        g.create_dataset("labels", data=y_test, dtype="float32")
        g.create_dataset("source_ids", data=sid_test, dtype="int64")


def save_h5_ood(
    output_h5: str,
    X_train,
    y_train,
    sid_train,
    X_val,
    y_val,
    sid_val,
    X_test_id,
    y_test_id,
    sid_test_id,
    X_test_ood,
    y_test_ood,
    sid_test_ood,
):
    os.makedirs(os.path.dirname(output_h5) or ".", exist_ok=True)

    with h5py.File(output_h5, "w") as f:
        g = f.create_group("train")
        g.create_dataset("spectrograms", data=X_train, dtype="float32")
        g.create_dataset("labels", data=y_train, dtype="float32")
        g.create_dataset("source_ids", data=sid_train, dtype="int64")

        g = f.create_group("val")
        g.create_dataset("spectrograms", data=X_val, dtype="float32")
        g.create_dataset("labels", data=y_val, dtype="float32")
        g.create_dataset("source_ids", data=sid_val, dtype="int64")

        g = f.create_group("test_id")
        g.create_dataset("spectrograms", data=X_test_id, dtype="float32")
        g.create_dataset("labels", data=y_test_id, dtype="float32")
        g.create_dataset("source_ids", data=sid_test_id, dtype="int64")

        g = f.create_group("test_ood")
        g.create_dataset("spectrograms", data=X_test_ood, dtype="float32")
        g.create_dataset("labels", data=y_test_ood, dtype="int64")
        g.create_dataset("source_ids", data=sid_test_ood, dtype="int64")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DIAT dataset to H5 with resize + sliding-window preprocessing "
                    "(classification or leave-one-class-out OOD)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/diat/DIAT-RadHAR",
        help="输入 DIAT 数据集目录",
    )
    parser.add_argument(
        "--output_h5",
        type=str,
        default="datasets/diat_processed.h5",
        help="输出 h5 文件路径",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=128,
        help="滑窗窗口大小，沿时间维切分",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=64,
        help="滑窗步幅，沿时间维移动",
    )
    parser.add_argument(
        "--holdout_class",
        type=int,
        default=5,
        help="可选：留作 OOD 的类别编号，范围 0~5；如果传 None，则执行普通 6 类分类",
    )

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.abspath(os.path.join(base_dir, args.input_dir))
    output_h5 = os.path.abspath(os.path.join(base_dir, args.output_h5))

    if args.holdout_class is None:
        print("[INFO] 运行模式: 普通6类分类")

        X_train_all, y_train_all, sid_train_all = [], [], []
        X_val_all, y_val_all, sid_val_all = [], [], []
        X_test_all, y_test_all, sid_test_all = [], [], []

        next_source_id = 0

        for class_idx, class_name in enumerate(DIAT_CLASSES):
            print(f"[INFO] 处理类别: {class_name}")

            X, y, source_ids, next_source_id = collect_class_samples(
                input_dir,
                class_name,
                class_idx,
                args.window_size,
                args.stride,
                next_source_id,
            )

            (
                X_tr,
                y_tr,
                sid_tr,
                X_va,
                y_va,
                sid_va,
                X_te,
                y_te,
                sid_te,
            ) = split_per_class(X, y, source_ids)

            X_train_all.append(X_tr)
            y_train_all.append(y_tr)
            sid_train_all.append(sid_tr)

            X_val_all.append(X_va)
            y_val_all.append(y_va)
            sid_val_all.append(sid_va)

            X_test_all.append(X_te)
            y_test_all.append(y_te)
            sid_test_all.append(sid_te)

            print(
                f"       窗口样本数: total={len(X)}, "
                f"train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}"
            )

        X_train = np.concatenate(X_train_all, axis=0)
        y_train = np.concatenate(y_train_all, axis=0)
        sid_train = np.concatenate(sid_train_all, axis=0)

        X_val = np.concatenate(X_val_all, axis=0)
        y_val = np.concatenate(y_val_all, axis=0)
        sid_val = np.concatenate(sid_val_all, axis=0)

        X_test = np.concatenate(X_test_all, axis=0)
        y_test = np.concatenate(y_test_all, axis=0)
        sid_test = np.concatenate(sid_test_all, axis=0)

        y_train_oh, y_val_oh, y_test_oh = one_hot_encode_labels(
            y_train,
            y_val,
            y_test,
            num_classes=len(DIAT_CLASSES),
        )

        print("\n[INFO] 最终数据形状:")
        print(f"  X_train: {X_train.shape}, y_train: {y_train_oh.shape}, source_ids: {sid_train.shape}")
        print(f"  X_val:   {X_val.shape}, y_val:   {y_val_oh.shape}, source_ids: {sid_val.shape}")
        print(f"  X_test:  {X_test.shape}, y_test:  {y_test_oh.shape}, source_ids: {sid_test.shape}")

        save_h5_classification(
            output_h5,
            X_train,
            y_train_oh,
            sid_train,
            X_val,
            y_val_oh,
            sid_val,
            X_test,
            y_test_oh,
            sid_test,
        )

        print(f"\n[OK] 已保存: {output_h5}")

    else:
        holdout_class = args.holdout_class
        if not (0 <= holdout_class < len(DIAT_CLASSES)):
            raise ValueError(f"--holdout_class 必须在 0~{len(DIAT_CLASSES) - 1} 之间")

        output_dir = os.path.dirname(output_h5)
        safe_class_name = make_safe_class_name(DIAT_CLASSES[holdout_class])
        output_h5 = os.path.join(output_dir, f"diat_processed_ood_{safe_class_name}.h5")

        print("[INFO] 运行模式: leave-one-class-out OOD")
        print(f"[INFO] OOD 类别编号: {holdout_class}")
        print(f"[INFO] OOD 类别名称: {DIAT_CLASSES[holdout_class]}")

        id_classes = [i for i in range(len(DIAT_CLASSES)) if i != holdout_class]
        label_map = {
            orig_label: new_label
            for new_label, orig_label in enumerate(id_classes)
        }

        X_train_all, y_train_all, sid_train_all = [], [], []
        X_val_all, y_val_all, sid_val_all = [], [], []
        X_test_id_all, y_test_id_all, sid_test_id_all = [], [], []
        X_test_ood_all, y_test_ood_all, sid_test_ood_all = [], [], []

        next_source_id = 0

        for class_idx, class_name in enumerate(DIAT_CLASSES):
            print(f"[INFO] 处理类别: {class_name}")

            X, y, source_ids, next_source_id = collect_class_samples(
                input_dir,
                class_name,
                class_idx,
                args.window_size,
                args.stride,
                next_source_id,
            )

            if class_idx == holdout_class:
                X_test_ood_all.append(X)
                y_test_ood_all.append(y)
                sid_test_ood_all.append(source_ids)

                print(f"       OOD窗口样本数: total={len(X)}")

            else:
                (
                    X_tr,
                    y_tr,
                    sid_tr,
                    X_va,
                    y_va,
                    sid_va,
                    X_te,
                    y_te,
                    sid_te,
                ) = split_per_class(X, y, source_ids)

                y_tr = np.asarray([label_map[v] for v in y_tr], dtype=np.int64)
                y_va = np.asarray([label_map[v] for v in y_va], dtype=np.int64)
                y_te = np.asarray([label_map[v] for v in y_te], dtype=np.int64)

                X_train_all.append(X_tr)
                y_train_all.append(y_tr)
                sid_train_all.append(sid_tr)

                X_val_all.append(X_va)
                y_val_all.append(y_va)
                sid_val_all.append(sid_va)

                X_test_id_all.append(X_te)
                y_test_id_all.append(y_te)
                sid_test_id_all.append(sid_te)

                print(
                    f"       ID窗口样本数: total={len(X)}, "
                    f"train={len(X_tr)}, val={len(X_va)}, test_id={len(X_te)}"
                )

        X_train = np.concatenate(X_train_all, axis=0)
        y_train = np.concatenate(y_train_all, axis=0)
        sid_train = np.concatenate(sid_train_all, axis=0)

        X_val = np.concatenate(X_val_all, axis=0)
        y_val = np.concatenate(y_val_all, axis=0)
        sid_val = np.concatenate(sid_val_all, axis=0)

        X_test_id = np.concatenate(X_test_id_all, axis=0)
        y_test_id = np.concatenate(y_test_id_all, axis=0)
        sid_test_id = np.concatenate(sid_test_id_all, axis=0)

        X_test_ood = np.concatenate(X_test_ood_all, axis=0)
        y_test_ood = np.concatenate(y_test_ood_all, axis=0)
        sid_test_ood = np.concatenate(sid_test_ood_all, axis=0)

        y_train_oh, y_val_oh, y_test_id_oh = one_hot_encode_labels(
            y_train,
            y_val,
            y_test_id,
            num_classes=len(id_classes),
        )

        print("\n[INFO] 最终数据形状:")
        print(f"  X_train:    {X_train.shape}, y_train:    {y_train_oh.shape}, source_ids: {sid_train.shape}")
        print(f"  X_val:      {X_val.shape}, y_val:      {y_val_oh.shape}, source_ids: {sid_val.shape}")
        print(f"  X_test_id:  {X_test_id.shape}, y_test_id:  {y_test_id_oh.shape}, source_ids: {sid_test_id.shape}")
        print(f"  X_test_ood: {X_test_ood.shape}, y_test_ood: {y_test_ood.shape}, source_ids: {sid_test_ood.shape}")

        save_h5_ood(
            output_h5,
            X_train,
            y_train_oh,
            sid_train,
            X_val,
            y_val_oh,
            sid_val,
            X_test_id,
            y_test_id_oh,
            sid_test_id,
            X_test_ood,
            y_test_ood,
            sid_test_ood,
        )

        print(f"\n[OK] 已保存: {output_h5}")
        print(f"[INFO] ID 类别编号: {id_classes}")


if __name__ == "__main__":
    main()
