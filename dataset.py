"""
dataset.py
----------
Dataset loader dan preprocessing untuk proyek Automated Essay Scoring (AES)
Multimodal: gambar tulisan tangan + teks OCR → skor esai

Sumber data:
  - ASAP Essay Scoring dataset (Kaggle): teks + label skor
  - IAM Handwriting DB / foto esai sendiri: gambar
"""

import os
import re
import ast
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import DistilBertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. KONFIGURASI GLOBAL
# ─────────────────────────────────────────────

class Config:
    # Path
    DATA_DIR       = Path("data")
    IMAGE_DIR      = DATA_DIR / "images"
    TEXT_DIR       = DATA_DIR / "texts"
    LABEL_CSV      = DATA_DIR / "labels.csv"
    CACHE_DIR      = DATA_DIR / ".cache"

    # Gambar
    IMAGE_SIZE     = (224, 224)      # input ResNet-50
    IMAGE_MEAN     = [0.485, 0.456, 0.406]   # ImageNet stats
    IMAGE_STD      = [0.229, 0.224, 0.225]

    # Teks
    TOKENIZER_NAME = "distilbert-base-uncased"
    MAX_SEQ_LEN    = 512

    # Dataset split
    TRAIN_RATIO    = 0.70
    VAL_RATIO      = 0.15
    TEST_RATIO     = 0.15
    RANDOM_SEED    = 42

    # DataLoader
    BATCH_SIZE     = 16
    NUM_WORKERS    = 4

    # ASAP prompt yang digunakan (1–8 tersedia)
    ESSAY_PROMPT   = 1            # ganti sesuai kebutuhan
    SCORE_MIN      = 2            # skor minimum prompt 1
    SCORE_MAX      = 12           # skor maksimum prompt 1


# ─────────────────────────────────────────────
# 2. PREPROCESSING TEKS
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Bersihkan teks hasil OCR dari noise umum:
    - Hapus karakter aneh / non-ASCII berlebih
    - Normalisasi spasi
    - Lowercase
    """
    # Hapus karakter non-printable
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    # Hapus angka isolasi (artefak OCR)
    text = re.sub(r'\b\d{1,2}\b', '', text)
    # Normalisasi spasi dan newline
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def normalize_score(score: float, s_min: float, s_max: float) -> float:
    """Normalisasi skor ke rentang [0, 1] untuk training."""
    return (score - s_min) / (s_max - s_min)


def denormalize_score(norm_score: float, s_min: float, s_max: float) -> float:
    """Kembalikan ke skala asli untuk evaluasi."""
    return norm_score * (s_max - s_min) + s_min


# ─────────────────────────────────────────────
# 3. TRANSFORMASI GAMBAR
# ─────────────────────────────────────────────

def get_image_transforms(mode: str = "train") -> transforms.Compose:
    """
    mode = 'train' : augmentasi aktif
    mode = 'val'   : hanya resize + normalize
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(Config.IMAGE_SIZE),
            transforms.RandomRotation(degrees=5),          # simulasi kemiringan tulisan
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1
            ),
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05)           # geser sedikit
            ),
            transforms.ToTensor(),
            transforms.Normalize(Config.IMAGE_MEAN, Config.IMAGE_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(Config.IMAGE_MEAN, Config.IMAGE_STD),
        ])


# ─────────────────────────────────────────────
# 4. DATASET CLASS UTAMA
# ─────────────────────────────────────────────

class EssayDataset(Dataset):
    """
    Dataset multimodal untuk essay scoring.

    Setiap item mengembalikan:
        image          : Tensor [3, 224, 224]
        input_ids      : Tensor [MAX_SEQ_LEN]
        attention_mask : Tensor [MAX_SEQ_LEN]
        score          : Tensor scalar (normalized)
        essay_id       : str (untuk debugging)
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        image_dir: Path,
        text_dir: Path,
        tokenizer: DistilBertTokenizer,
        transform: Optional[transforms.Compose] = None,
        use_cached_text: bool = True,
    ):
        self.df          = labels_df.reset_index(drop=True)
        self.image_dir   = Path(image_dir)
        self.text_dir    = Path(text_dir)
        self.tokenizer   = tokenizer
        self.transform   = transform or get_image_transforms("val")
        self.cache       = {}
        self.use_cached  = use_cached_text

        self._validate_paths()

    def _validate_paths(self):
        missing = []
        for _, row in self.df.iterrows():
            img_path = self.image_dir / f"{row['essay_id']}.jpg"
            txt_path = self.text_dir  / f"{row['essay_id']}.txt"
            if not img_path.exists():
                missing.append(str(img_path))
            if not txt_path.exists():
                missing.append(str(txt_path))
        if missing:
            logger.warning(f"{len(missing)} file tidak ditemukan. "
                           f"Contoh: {missing[:3]}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, essay_id: str) -> Image.Image:
        path = self.image_dir / f"{essay_id}.jpg"
        try:
            img = Image.open(path).convert("RGB")
            # Binarisasi ringan untuk memperjelas tulisan tangan
            img = ImageOps.autocontrast(img)
            return img
        except FileNotFoundError:
            logger.warning(f"Gambar tidak ditemukan: {path}. Diganti blank.")
            return Image.new("RGB", Config.IMAGE_SIZE, color=(255, 255, 255))

    def _load_text(self, essay_id: str) -> str:
        if self.use_cached and essay_id in self.cache:
            return self.cache[essay_id]
        path = self.text_dir / f"{essay_id}.txt"
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(f"Teks tidak ditemukan: {path}. Diganti string kosong.")
            text = ""
        text = clean_text(text)
        if self.use_cached:
            self.cache[essay_id] = text
        return text

    def __getitem__(self, idx: int) -> dict:
        row      = self.df.iloc[idx]
        essay_id = str(row["essay_id"])
        score    = float(row["normalized_score"])

        # --- Gambar ---
        image = self._load_image(essay_id)
        image = self.transform(image)

        # --- Teks ---
        text    = self._load_text(essay_id)
        encoded = self.tokenizer(
            text,
            max_length     = Config.MAX_SEQ_LEN,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        input_ids      = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "image"          : image,
            "input_ids"      : input_ids,
            "attention_mask" : attention_mask,
            "score"          : torch.tensor(score, dtype=torch.float32),
            "essay_id"       : essay_id,
        }


# ─────────────────────────────────────────────
# 5. LOAD DAN SPLIT DATASET
# ─────────────────────────────────────────────

def load_asap_labels(
    csv_path: Path,
    prompt_id: int = 1,
    score_col: str = "domain1_score",
) -> pd.DataFrame:
    """
    Load file TSV ASAP dari Kaggle (training_set_rel3.tsv).
    Filter berdasarkan essay_set (prompt_id).
    Tambahkan kolom normalized_score.
    """
    df = pd.read_csv(csv_path, sep="\t", encoding="latin-1")

    # Filter prompt
    df = df[df["essay_set"] == prompt_id].copy()
    df = df.rename(columns={"essay_id": "essay_id", score_col: "score"})

    # Normalisasi skor
    s_min = Config.SCORE_MIN
    s_max = Config.SCORE_MAX
    df["normalized_score"] = df["score"].apply(
        lambda s: normalize_score(s, s_min, s_max)
    )

    logger.info(
        f"Prompt {prompt_id}: {len(df)} esai | "
        f"Skor {s_min}–{s_max} | "
        f"Rata-rata: {df['score'].mean():.2f}"
    )
    return df[["essay_id", "score", "normalized_score"]]


def load_custom_labels(csv_path: Path) -> pd.DataFrame:
    """
    Alternatif: load label dari CSV buatan sendiri.
    Format CSV: essay_id, score
    """
    df = pd.read_csv(csv_path)
    assert "essay_id" in df.columns, "CSV harus punya kolom 'essay_id'"
    assert "score"    in df.columns, "CSV harus punya kolom 'score'"
    s_min = df["score"].min()
    s_max = df["score"].max()
    df["normalized_score"] = df["score"].apply(
        lambda s: normalize_score(s, s_min, s_max)
    )
    logger.info(f"Custom dataset: {len(df)} esai | skor {s_min}–{s_max}")
    return df


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float = Config.TRAIN_RATIO,
    val_ratio:   float = Config.VAL_RATIO,
    seed:        int   = Config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split stratified berdasarkan skor (bin ke 5 kelompok)."""
    from sklearn.model_selection import train_test_split

    # Stratifikasi berdasarkan bin skor
    df["score_bin"] = pd.cut(df["score"], bins=5, labels=False)

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio),
        stratify=df["score_bin"], random_state=seed
    )
    val_ratio_adjusted = val_ratio / (val_ratio + Config.TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_adjusted),
        stratify=temp_df["score_bin"], random_state=seed
    )

    logger.info(
        f"Split → Train: {len(train_df)} | "
        f"Val: {len(val_df)} | Test: {len(test_df)}"
    )
    return (
        train_df.drop(columns="score_bin"),
        val_df.drop(columns="score_bin"),
        test_df.drop(columns="score_bin"),
    )


# ─────────────────────────────────────────────
# 6. DATALOADER FACTORY
# ─────────────────────────────────────────────

def build_dataloaders(
    label_csv:  Path = Config.LABEL_CSV,
    image_dir:  Path = Config.IMAGE_DIR,
    text_dir:   Path = Config.TEXT_DIR,
    batch_size: int  = Config.BATCH_SIZE,
    use_asap:   bool = True,
) -> dict[str, DataLoader]:
    """
    Buat train/val/test DataLoader sekaligus.

    Returns:
        {
          'train': DataLoader,
          'val':   DataLoader,
          'test':  DataLoader,
          'config': {'score_min', 'score_max'}
        }
    """
    # Load label
    if use_asap:
        df = load_asap_labels(label_csv, prompt_id=Config.ESSAY_PROMPT)
    else:
        df = load_custom_labels(label_csv)

    train_df, val_df, test_df = split_dataframe(df)

    # Tokenizer (shared)
    tokenizer = DistilBertTokenizer.from_pretrained(Config.TOKENIZER_NAME)

    def make_loader(split_df, mode):
        ds = EssayDataset(
            labels_df  = split_df,
            image_dir  = image_dir,
            text_dir   = text_dir,
            tokenizer  = tokenizer,
            transform  = get_image_transforms(mode),
        )
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (mode == "train"),
            num_workers = Config.NUM_WORKERS,
            pin_memory  = True,
        )

    return {
        "train"  : make_loader(train_df, "train"),
        "val"    : make_loader(val_df,   "val"),
        "test"   : make_loader(test_df,  "val"),
        "config" : {"score_min": Config.SCORE_MIN, "score_max": Config.SCORE_MAX},
    }


# ─────────────────────────────────────────────
# 7. UTILITAS EKSPLORASI DATA
# ─────────────────────────────────────────────

def describe_dataset(df: pd.DataFrame) -> None:
    """Tampilkan statistik dasar dataset."""
    print("=" * 45)
    print(f"  Jumlah sampel : {len(df)}")
    print(f"  Skor rata-rata: {df['score'].mean():.2f}")
    print(f"  Skor std      : {df['score'].std():.2f}")
    print(f"  Distribusi    :")
    print(df["score"].value_counts().sort_index().to_string())
    print("=" * 45)


def visualize_sample(dataset: EssayDataset, idx: int = 0) -> None:
    """Tampilkan satu sampel untuk verifikasi."""
    import matplotlib.pyplot as plt

    sample   = dataset[idx]
    img_np   = sample["image"].permute(1, 2, 0).numpy()
    # De-normalize untuk display
    mean = np.array(Config.IMAGE_MEAN)
    std  = np.array(Config.IMAGE_STD)
    img_np = (img_np * std + mean).clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(img_np)
    axes[0].set_title(f"Essay ID: {sample['essay_id']}")
    axes[0].axis("off")

    # Decode token untuk display
    tokenizer = DistilBertTokenizer.from_pretrained(Config.TOKENIZER_NAME)
    tokens = tokenizer.decode(
        sample["input_ids"], skip_special_tokens=True
    )
    axes[1].text(
        0.01, 0.99, tokens[:300] + "...",
        transform=axes[1].transAxes,
        va="top", wrap=True, fontsize=9
    )
    axes[1].set_title(
        f"Normalized score: {sample['score']:.3f} "
        f"({denormalize_score(sample['score'].item(), Config.SCORE_MIN, Config.SCORE_MAX):.1f} raw)"
    )
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(f"sample_{idx}.png", dpi=100)
    plt.show()


# ─────────────────────────────────────────────
# 8. QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Simulasi cepat tanpa file nyata
    import tempfile

    print("Menguji dataset loader dengan data dummy...")

    # Buat DataFrame dummy
    np.random.seed(42)
    n = 50
    dummy_df = pd.DataFrame({
        "essay_id"         : [f"essay_{i:04d}" for i in range(n)],
        "score"            : np.random.randint(2, 13, n),
        "normalized_score" : np.random.uniform(0, 1, n),
    })

    tokenizer = DistilBertTokenizer.from_pretrained(Config.TOKENIZER_NAME)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "images"
        txt_dir = Path(tmpdir) / "texts"
        img_dir.mkdir(); txt_dir.mkdir()

        # Buat file dummy
        for eid in dummy_df["essay_id"]:
            Image.new("RGB", (300, 400), color=(245, 245, 245)).save(
                img_dir / f"{eid}.jpg"
            )
            (txt_dir / f"{eid}.txt").write_text(
                "This is a sample essay about the importance of education "
                "and how students can improve their writing skills.", "utf-8"
            )

        ds = EssayDataset(
            labels_df  = dummy_df,
            image_dir  = img_dir,
            text_dir   = txt_dir,
            tokenizer  = tokenizer,
            transform  = get_image_transforms("train"),
        )

        sample = ds[0]
        print(f"  image shape     : {sample['image'].shape}")
        print(f"  input_ids shape : {sample['input_ids'].shape}")
        print(f"  score           : {sample['score']:.4f}")
        print(f"  essay_id        : {sample['essay_id']}")
        print("\nDataset loader OK!")

from torchvision import transforms
from transformers import AutoTokenizer

def get_transforms_and_tokenizer(model_name: str = "bert-base-uncased"):
    """
    Fungsi jembatan untuk memuat Tokenizer dan Transformasi Gambar 
    yang konsisten antara tahap training (Colab) dan produksi (Streamlit).
    """
    # 1. Definisikan transformasi gambar standar (Suaikan dengan ukuran input model Anda, misal 224x224)
    transform_fn = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # Standar ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 2. Definisikan Tokenizer dari Hugging Face
    # Ganti "bert-base-uncased" dengan nama model teks yang Anda gunakan saat training
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return tokenizer, transform_fn
