import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import random
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
#  SEED
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─────────────────────────────────────────────
#  DATASET
#  Her dosya bir pencere (0.8 sn) : tek ornek
#  Etiket: combined = person * 2 + target
#    (person=0, target=0) : sinif 0
#    (person=0, target=1) : sinif 1
#    (person=1, target=0) : sinif 2
#    (person=1, target=1) : sinif 3
#  Boylece [0,1] ve [1,1] kesinlikle farkli siniflar.
# ─────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(self, folderpath, device='cpu', mean=None, std=None):
        super().__init__()
        self.folderpath = folderpath
        self.files = os.listdir(folderpath)
        self.mean = mean
        self.std = std
        self.device = device
        self.bands = [
            (0.5, 4),
            (4, 8),
            (8, 13),
            (13, 30),
            (30, 45),
        ]
        self.band_rate = 512

    def bandpass_filter(self, data, low, high, fs, order=4):
        nyquist = 0.5 * fs
        low  = low  / nyquist
        high = high / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = os.path.join(self.folderpath, self.files[index])
        eeg  = pd.read_csv(path).to_numpy()

        base_name = self.files[index].replace(".csv", "")
        parts  = base_name.split("_")
        person = int(parts[1])
        target = int(parts[2])

        # Her frekans bandi icin bandpass filtrele
        band_data = []
        for low, high in self.bands:
            filtered = self.bandpass_filter(eeg, low, high, self.band_rate)
            band_data.append(filtered)
        band_data = np.stack(band_data, axis=0)  # (bands, elektrod, zaman)

        # Global istatistiklerle normalize et
        if self.mean is not None and self.std is not None:
            band_data = (band_data - self.mean) / (self.std + 1e-8)

        x = torch.tensor(band_data, dtype=torch.float32, device=self.device)

        # y bilerek hem person hem target etiketine gore ayarlandi.
        # Onceki calismalar gosteriyor ki kisi bagimsiz, sadece uyarana
        # bagli degismeyen bir pattern mevcut. Sadece targeta bagli egitimde
        # CNN bu patternin yaninda kisi-bagimsiz patterni de ogrenebilir.
        # Bunu onlemek icin CNN egitiminde her (person, target) ciftini
        # ayri bir sinif olarak veriyoruz → combined label.
        combined = person * 2 + target
        y = torch.tensor(combined, dtype=torch.long, device=self.device)

        return x, y


def compute_global_stats(dataset):
    """
    Train dataseti uzerinden global mean ve std hesapla.
    Normalizasyon icin sadece train'e fit et.
    """
    sum_   = 0.0
    sum_sq = 0.0
    count  = 0
    for i in range(len(dataset)):
        x, _ = dataset[i]
        x = x.numpy()
        sum_   += x.sum()
        sum_sq += (x ** 2).sum()
        count  += x.size
    mean = sum_ / count
    var  = sum_sq / count - mean ** 2
    std  = np.sqrt(var)
    return mean, std


# ─────────────────────────────────────────────
#  MODEL
#  Giris : (batch, bands=5, elektrod, zaman)
#  Cikis : (batch, 4)  →  CrossEntropyLoss
#
#  CatBoost icin: extract_features() kullan,
#  head atlanir, (batch, feature_dim) vektoru alinir.
# ─────────────────────────────────────────────

class EEGNet(nn.Module):
    def __init__(
        self,
        n_bands: int,
        n_electrodes: int,
        feature_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()

        # ── 1. Band-wise Temporal Conv ──────────────────────────────
        # groups=n_bands → her frekans bandi bagimsiz filtre ogrenir
        self.band_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=n_bands,
                out_channels=n_bands * 8,
                kernel_size=(1, 64),
                padding=(0, 32),
                groups=n_bands,
                bias=False,
            ),
            nn.BatchNorm2d(n_bands * 8),
            nn.ELU(),
            nn.Dropout2d(dropout * 0.5),
        )

        # ── 2. Spatial Conv ─────────────────────────────────────────
        # kernel (n_electrodes, 1) → tum elektrodlari birlestir
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=n_bands * 8,
                out_channels=n_bands * 16,
                kernel_size=(n_electrodes, 1),
                bias=False,
            ),
            nn.BatchNorm2d(n_bands * 16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout2d(dropout * 0.5),
        )

        # ── 3. Temporal Separable Conv ──────────────────────────────
        spatial_out_ch = n_bands * 16
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=spatial_out_ch,
                out_channels=spatial_out_ch,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=spatial_out_ch,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=spatial_out_ch,
                out_channels=feature_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout),
        )

        # ── 4. Classifier head (CatBoost icin kaldirilacak) ─────────
        # 4 sinif: her (person, target) kombinasyonu ayri sinif
        self.head = nn.Linear(feature_dim, 4)

    def extract_features(self, x):
        """
        CatBoost icin feature vektoru cikar.
        x      : (batch, bands, elektrod, zaman)
        return : (batch, feature_dim)
        """
        x = self.band_conv(x)
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        return x.flatten(1)

    def forward(self, x):
        """
        return: (batch, 4)
        """
        return self.head(self.extract_features(x))


# ─────────────────────────────────────────────
#  EGITIM DONGUSU
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total = 0.0, 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total      += x.size(0)

    return {"loss": total_loss / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        total_loss += criterion(model(x), y).item() * x.size(0)
        total      += x.size(0)

    return {"loss": total_loss / total}


def train(model,train_loader,test_loader,device,n_epochs: int = 50,lr: float = 1e-3,save_dir: str = "checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float("inf")
    history = []

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8}")
    print("-" * 30)

    for epoch in range(1, n_epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch":      epoch,
            "train_loss": train_m["loss"],
            "val_loss":   val_m["loss"],
        })

        print(f"{epoch:>6} | {train_m['loss']:>10.4f} | {val_m['loss']:>8.4f}")

        # Her epoch sonunda checkpoint kaydet
        ckpt = {
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics":       train_m,
            "val_metrics":         val_m,
        }
        torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch:03d}.pt"))

        # En iyi modeli ayrica kaydet
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save(ckpt, os.path.join(save_dir, "best_model.pt"))
            print(f"         -> en iyi model kaydedildi (val_loss={best_val_loss:.4f})")

    return history


# ─────────────────────────────────────────────
#  CATBOOST ICIN FEATURE CIKARMA
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_features_for_catboost(model, loader, device):
    """
    Modelin classifier kafasi olmadan feature vektoru dondurur.
    return:
        features : np.ndarray (N, feature_dim)
        labels   : np.ndarray (N,) combined label
    """
    model.eval()
    all_features, all_labels = [], []

    for x, y in loader:
        feat = model.extract_features(x.to(device))
        all_features.append(feat.cpu().numpy())
        all_labels.append(y.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


# ─────────────────────────────────────────────
#  ANA AKIS
# ─────────────────────────────────────────────

if __name__ == "__main__":

    BATCH_SIZE = 32
    NUM_WORKERS = 2
    TRAINING = True

    # ── Dataset ve split ──────────────────────
    all_files = os.listdir("data_files")

    split_labels = []

    for f in all_files:
        name = f.replace(".csv", "")
        parts = name.split("_")
        person = int(parts[1])
        target = int(parts[2])
        combined = person * 2 + target
        split_labels.append(combined)

    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=SEED,stratify=split_labels)
    del split_labels

    train_dataset = EEGDataset("data_files", device='cpu')
    train_dataset.files = train_files
    test_dataset  = EEGDataset("data_files", device='cpu')
    test_dataset.files  = test_files

    # Global istatistikleri sadece train uzerinden hesapla
    mean, std = compute_global_stats(train_dataset)
    train_dataset.mean, train_dataset.std = mean, std
    test_dataset.mean,  test_dataset.std  = mean, std

    

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS, pin_memory=(device == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS, pin_memory=(device == 'cuda'))

    # data_files2: hic gorulmemis gercek test seti
    # Ayni mean/std kullaniliyor → train dagitimindan normalize et
    different_dataset = EEGDataset("data_files2", device='cpu')
    different_dataset.mean = mean
    different_dataset.std  = std
    different_loader = DataLoader(different_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS, pin_memory=(device == 'cuda'))

    # ── Model ─────────────────────────────────
    sample_x, _ = next(iter(train_loader))
    _, n_bands, n_electrodes, _ = sample_x.shape

    print(f"Giris: bands={n_bands}, elektrod={n_electrodes}")

    model = EEGNet(
        n_bands=n_bands,
        n_electrodes=n_electrodes,
        feature_dim=256,
        dropout=0.5,
    ).to(device)

    print(f"Parametre sayisi: {sum(p.numel() for p in model.parameters()):,}")

    # ── Egitim ────────────────────────────────
    if TRAINING : 
        history = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            n_epochs=50,
            lr=1e-3,
            save_dir="checkpoints",
        )

    # ── En iyi modeli yukle ───────────────────
    ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── CatBoost icin feature cikar ───────────
    train_feat, train_lbl = extract_features_for_catboost(model, train_loader,    device)
    test_feat,  test_lbl  = extract_features_for_catboost(model, test_loader,     device)
    diff_feat,  diff_lbl  = extract_features_for_catboost(model, different_loader, device)

    # combined label'i geri coz:
    # y_person = lbl // 2  →  0 veya 1
    # y_target = lbl % 2   →  0 veya 1

    np.save("train_features.npy",    train_feat)
    np.save("train_labels.npy",      train_lbl)
    np.save("test_features.npy",     test_feat)
    np.save("test_labels.npy",       test_lbl)
    np.save("different_features.npy", diff_feat)
    np.save("different_labels.npy",   diff_lbl)

    print(f"Feature boyutu: {train_feat.shape}")
    print("Kaydedildi.")