'''
Bu kod en degisigi cunku target ve kisi bagimsiz ogreniyor iki asamali dogrulama icin en uygunu bu kisinin unique degerini aliyor (session bagimsiz)
ve bunu targetin kisi ve session bagimsiz bilgisi ile birlestiriyor bu sekilde kisiyi target ile birlestirerek guclu omasini bekliyorum 

'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import random
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.autograd import Function

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
#
#  Egitim verisi : id_kisi_target_session.csv
#  Test verisi   : id_kisi_target.csv (session yok)
# ─────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(self, folderpath, device='cpu', mean=None, std=None, has_session=True):
        super().__init__()
        self.folderpath  = folderpath
        self.files       = os.listdir(folderpath)
        self.mean        = mean
        self.std         = std
        self.device      = device
        self.has_session = has_session
        self.bands       = [
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
        parts   = base_name.split("_")
        person  = int(parts[1])
        target  = int(parts[2])
        session = int(parts[3]) - 1 if self.has_session and len(parts) > 3 else 0

        band_data = []
        for low, high in self.bands:
            filtered = self.bandpass_filter(eeg, low, high, self.band_rate)
            band_data.append(filtered)
        band_data = np.stack(band_data, axis=0)  # (bands, elektrod, zaman)

        if self.mean is not None and self.std is not None:
            band_data = (band_data - self.mean) / (self.std + 1e-8)

        x         = torch.tensor(band_data, dtype=torch.float32, device=self.device)
        y_person  = torch.tensor(person,    dtype=torch.long,    device=self.device)
        y_target  = torch.tensor(target,    dtype=torch.long,    device=self.device)
        y_session = torch.tensor(session,   dtype=torch.long,    device=self.device)

        return x, y_person, y_target, y_session


def compute_global_stats(dataset):
    sum_   = 0.0
    sum_sq = 0.0
    count  = 0
    for i in range(len(dataset)):
        x, _, _, _ = dataset[i]
        x = x.numpy()
        sum_   += x.sum()
        sum_sq += (x ** 2).sum()
        count  += x.size
    mean = sum_ / count
    var  = sum_sq / count - mean ** 2
    std  = np.sqrt(var)
    return mean, std


# ─────────────────────────────────────────────
#  GRADIENT REVERSAL LAYER
# ─────────────────────────────────────────────

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# ─────────────────────────────────────────────
#  MODEL
#
#  Paylasilan omurga → feature_dim boyutlu vektor
#
#  Buradan iki AYRI projeksiyon:
#    person_proj (proj_dim) → kisi bilgisi
#                             triplet: ayni kisi farkli session yakin
#                                      farkli kisi uzak (target farketmez)
#
#    target_proj (proj_dim) → uyaran bilgisi
#                             triplet: ayni target farkli session yakin
#                                      farkli target uzak (kisi farketmez)
#
#  Classification headlari:
#    person_head  → tum veri
#    target_head  → SADECE genuine ornekler (person=1)
#    session_head → GRL ile session baskilanir (her iki projeksiyona da uygulanir)
#
#  CatBoost feature = concat(person_proj, target_proj) → proj_dim * 2
# ─────────────────────────────────────────────

class EEGNet(nn.Module):
    def __init__(
        self,
        n_bands: int,
        n_electrodes: int,
        n_sessions: int,
        feature_dim: int = 256,
        proj_dim: int = 128,
        dropout: float = 0.5,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # ── 1. Paylasilan omurga ────────────────────────────────────

        # Band-wise Temporal Conv
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

        # Spatial Conv: tum elektrodlari birlestir
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

        # Temporal Separable Conv
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

        # ── 2. Iki ayri projeksiyon ─────────────────────────────────

        # Kisi projeksiyonu: kisi bilgisini target'tan bagimsiz encode eder
        self.person_proj = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ELU(),
        )

        # Target projeksiyonu: uyaran bilgisini kisiden bagimsiz encode eder
        self.target_proj = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ELU(),
        )

        # ── 3. Classification headlari ──────────────────────────────

        # Person head: tum veri, kisi projeksiyonu uzerinden
        self.person_head = nn.Linear(proj_dim, 2)

        # Target head: SADECE genuine ornekler, target projeksiyonu uzerinden
        self.target_head = nn.Linear(proj_dim, 2)

        # ── 4. Session adversary headlari ───────────────────────────
        # Her iki projeksiyona ayri GRL uygulanir
        # Boylece her iki projeksiyon da session bilgisini gizlemeye zorlanir
        self.grl_person          = GradientReversalLayer(lambda_=grl_lambda)
        self.grl_target          = GradientReversalLayer(lambda_=grl_lambda)
        self.session_head_person = nn.Linear(proj_dim, n_sessions)
        self.session_head_target = nn.Linear(proj_dim, n_sessions)

    def extract_backbone(self, x):
        """
        Paylasilan omurgadan ham feature vektoru cikar.
        x      : (batch, bands, elektrod, zaman)
        return : (batch, feature_dim)
        """
        x = self.band_conv(x)
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        return x.flatten(1)

    def extract_features(self, x):
        """
        CatBoost icin iki projeksiyonu concat ederek dondurur.
        return : (batch, proj_dim * 2)
        """
        feat        = self.extract_backbone(x)
        p_feat      = self.person_proj(feat)
        t_feat      = self.target_proj(feat)
        return torch.cat([p_feat, t_feat], dim=1)

    def forward(self, x):
        """
        return:
            person_logits         : (batch, 2)
            target_logits         : (batch, 2)
            session_logits_person : (batch, n_sessions)
            session_logits_target : (batch, n_sessions)
            p_feat                : (batch, proj_dim)  person triplet icin
            t_feat                : (batch, proj_dim)  target triplet icin
        """
        feat    = self.extract_backbone(x)
        p_feat  = self.person_proj(feat)
        t_feat  = self.target_proj(feat)

        person_out          = self.person_head(p_feat)
        target_out          = self.target_head(t_feat)
        session_out_person  = self.session_head_person(self.grl_person(p_feat))
        session_out_target  = self.session_head_target(self.grl_target(t_feat))

        return (person_out, target_out,
                session_out_person, session_out_target,
                p_feat, t_feat)


# ─────────────────────────────────────────────
#  TRIPLET LOSS - PERSON
#  Positive: ayni kisi, farkli session (target farketmez)
#  Negative: farkli kisi (session ve target farketmez)
#  Boylece kisi bilgisi target'tan BAGIMSIZ ogrenilir
# ─────────────────────────────────────────────

def triplet_loss_person(p_feat, person_labels, session_labels, margin=1.0):
    p_feat   = F.normalize(p_feat, p=2, dim=1)
    dist_mat = torch.cdist(p_feat, p_feat, p=2)

    loss, n_triplets = torch.tensor(0.0, device=p_feat.device), 0

    for i in range(p_feat.size(0)):
        p_i = person_labels[i]
        s_i = session_labels[i]

        # Positive: ayni kisi, farkli session, target farketmez
        pos_mask = (person_labels == p_i) & (session_labels != s_i)
        # Negative: farkli kisi, her sey farketmez
        neg_mask = (person_labels != p_i)

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        pos_dist    = dist_mat[i][pos_mask].max()
        neg_dist    = dist_mat[i][neg_mask].min()
        loss       += F.relu(pos_dist - neg_dist + margin)
        n_triplets += 1

    return loss / n_triplets if n_triplets > 0 else torch.tensor(0.0, device=p_feat.device)


# ─────────────────────────────────────────────
#  TRIPLET LOSS - TARGET
#  Positive: ayni target, farkli session (kisi farketmez)
#  Negative: farkli target (session ve kisi farketmez)
#  Boylece target bilgisi kisiden BAGIMSIZ ogrenilir
# ─────────────────────────────────────────────

def triplet_loss_target(t_feat, target_labels, session_labels, margin=1.0):
    t_feat   = F.normalize(t_feat, p=2, dim=1)
    dist_mat = torch.cdist(t_feat, t_feat, p=2)

    loss, n_triplets = torch.tensor(0.0, device=t_feat.device), 0

    for i in range(t_feat.size(0)):
        t_i = target_labels[i]
        s_i = session_labels[i]

        # Positive: ayni target, farkli session, kisi farketmez
        pos_mask = (target_labels == t_i) & (session_labels != s_i)
        # Negative: farkli target, her sey farketmez
        neg_mask = (target_labels != t_i)

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        pos_dist    = dist_mat[i][pos_mask].max()
        neg_dist    = dist_mat[i][neg_mask].min()
        loss       += F.relu(pos_dist - neg_dist + margin)
        n_triplets += 1

    return loss / n_triplets if n_triplets > 0 else torch.tensor(0.0, device=t_feat.device)


# ─────────────────────────────────────────────
#  EGITIM DONGUSU
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, alpha, beta, gamma):
    model.train()
    total_loss, total = 0.0, 0

    for x, y_person, y_target, y_session in loader:
        x         = x.to(device)
        y_person  = y_person.to(device)
        y_target  = y_target.to(device)
        y_session = y_session.to(device)

        optimizer.zero_grad()

        (person_out, target_out,
         session_out_p, session_out_t,
         p_feat, t_feat) = model(x)

        # 1. Person loss: tum ornekler, kisi projeksiyonu
        loss_person = criterion(person_out, y_person)

        # 2. Target loss: SADECE genuine ornekler, target projeksiyonu
        genuine_mask = (y_person == 1)
        if genuine_mask.sum() > 0:
            loss_target = criterion(
                target_out[genuine_mask],
                y_target[genuine_mask]
            )
        else:
            loss_target = torch.tensor(0.0, device=device)

        # 3. Triplet person: kisi bilgisi target bagimsiz, session invariant
        loss_trip_p = triplet_loss_person(p_feat, y_person, y_session)

        # 4. Triplet target: uyaran bilgisi kisi bagimsiz, session invariant
        loss_trip_t = triplet_loss_target(t_feat, y_target, y_session)

        # 5. Session adversary: her iki projeksiyon icin ayri GRL
        loss_session = (criterion(session_out_p, y_session) +
                        criterion(session_out_t, y_session))

        loss = (loss_person
                + alpha * loss_target
                + beta  * loss_trip_p
                + beta  * loss_trip_t
                + gamma * loss_session)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total      += x.size(0)

    return {"loss": total_loss / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device, alpha, beta, gamma):
    model.eval()
    total_loss, total = 0.0, 0

    for x, y_person, y_target, y_session in loader:
        x         = x.to(device)
        y_person  = y_person.to(device)
        y_target  = y_target.to(device)
        y_session = y_session.to(device)

        (person_out, target_out,
         session_out_p, session_out_t,
         p_feat, t_feat) = model(x)

        loss_person = criterion(person_out, y_person)

        genuine_mask = (y_person == 1)
        if genuine_mask.sum() > 0:
            loss_target = criterion(
                target_out[genuine_mask],
                y_target[genuine_mask]
            )
        else:
            loss_target = torch.tensor(0.0, device=device)

        loss_trip_p  = triplet_loss_person(p_feat, y_person, y_session)
        loss_trip_t  = triplet_loss_target(t_feat, y_target, y_session)
        loss_session = (criterion(session_out_p, y_session) +
                        criterion(session_out_t, y_session))

        loss = (loss_person
                + alpha * loss_target
                + beta  * loss_trip_p
                + beta  * loss_trip_t
                + gamma * loss_session)

        total_loss += loss.item() * x.size(0)
        total      += x.size(0)

    return {"loss": total_loss / total}


def train(
    model,
    train_loader,
    test_loader,
    device,
    n_epochs: int = 50,
    lr: float = 1e-3,
    save_dir: str = "checkpoints",
    alpha: float = 0.5,   # target loss agirligi
    beta: float  = 0.5,   # triplet loss agirligi (her ikisi icin)
    gamma: float = 0.3,   # session adversary agirligi
):
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float("inf")
    history       = []

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8}")
    print("-" * 30)

    for epoch in range(1, n_epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            alpha, beta, gamma
        )
        val_m = evaluate(
            model, test_loader, criterion, device,
            alpha, beta, gamma
        )
        scheduler.step()

        history.append({
            "epoch":      epoch,
            "train_loss": train_m["loss"],
            "val_loss":   val_m["loss"],
        })

        print(f"{epoch:>6} | {train_m['loss']:>10.4f} | {val_m['loss']:>8.4f}")

        ckpt = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics":        train_m,
            "val_metrics":          val_m,
        }
        torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch:03d}.pt"))

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save(ckpt, os.path.join(save_dir, "best_model.pt"))
            print(f"         -> en iyi model kaydedildi (val_loss={best_val_loss:.4f})")

    return history


# ─────────────────────────────────────────────
#  CATBOOST ICIN FEATURE CIKARMA
#  CatBoost feature = concat(person_proj, target_proj)
#  person_proj → kisi bilgisi (target bagimsiz, session invariant)
#  target_proj → uyaran bilgisi (kisi bagimsiz, session invariant)
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_features_for_catboost(model, loader, device):
    """
    return:
        features : np.ndarray (N, proj_dim * 2)
        y_person : np.ndarray (N,)
        y_target : np.ndarray (N,)
    """
    model.eval()
    all_features = []
    all_person   = []
    all_target   = []

    for x, y_person, y_target, _ in loader:
        feat = model.extract_features(x.to(device))
        all_features.append(feat.cpu().numpy())
        all_person.append(y_person.numpy())
        all_target.append(y_target.numpy())

    return (
        np.concatenate(all_features),
        np.concatenate(all_person),
        np.concatenate(all_target),
    )


# ─────────────────────────────────────────────
#  CATBOOST IKI ASAMALI PIPELINE
# ─────────────────────────────────────────────

def two_stage_predict(person_model, target_model, x):
    person_pred = person_model.predict(x)
    target_pred = np.zeros(len(x), dtype=int)

    genuine_mask = (person_pred == 1)
    if genuine_mask.sum() > 0:
        target_pred[genuine_mask] = target_model.predict(x[genuine_mask])

    return person_pred, target_pred


def pipeline_score(y_true_person, y_true_target, person_pred, target_pred):
    from sklearn.metrics import classification_report

    print("=== PERSON ===")
    print(classification_report(y_true_person, person_pred))

    genuine_mask = (person_pred == 1)
    if genuine_mask.sum() > 0:
        print("=== TARGET (genuine tahmin edilenlerde) ===")
        print(classification_report(
            y_true_target[genuine_mask],
            target_pred[genuine_mask]
        ))

    y_true_combined = ((y_true_person == 1) & (y_true_target == 1)).astype(int)
    y_pred_combined = ((person_pred   == 1) & (target_pred   == 1)).astype(int)
    print("=== PIPELINE TOPLAM (person=1 VE target=1) ===")
    print(classification_report(y_true_combined, y_pred_combined))


# ─────────────────────────────────────────────
#  ANA AKIS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    #from catboost import CatBoostClassifier

    # ── Dataset ───────────────────────────────
    all_files = os.listdir("data_files_session")
    train_files, test_files = train_test_split(
        all_files, test_size=0.2, random_state=SEED
    )

    train_dataset = EEGDataset("data_files_session", device='cpu', has_session=True)
    train_dataset.files = train_files
    test_dataset  = EEGDataset("data_files_session", device='cpu', has_session=True)
    test_dataset.files  = test_files

    mean, std = compute_global_stats(train_dataset)
    train_dataset.mean, train_dataset.std = mean, std
    test_dataset.mean,  test_dataset.std  = mean, std

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=2, pin_memory=(device == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=2, pin_memory=(device == 'cuda')
    )

    # 8. session: session bilgisi yok
    different_dataset = EEGDataset("../data_files2", device='cpu', has_session=False)
    different_dataset.mean = mean
    different_dataset.std  = std
    different_loader = DataLoader(
        different_dataset, batch_size=32, shuffle=False,
        num_workers=2, pin_memory=(device == 'cuda')
    )

    # ── Model boyutlari ───────────────────────
    sample_x, _, _, _ = next(iter(train_loader))
    _, n_bands, n_electrodes, _ = sample_x.shape

    n_sessions = len(set(
        int(f.replace(".csv", "").split("_")[3]) for f in all_files
    ))

    print(f"Giris       : bands={n_bands}, elektrod={n_electrodes}")
    print(f"Session     : {n_sessions}")

    model = EEGNet(
        n_bands=n_bands,
        n_electrodes=n_electrodes,
        n_sessions=n_sessions,
        feature_dim=256,
        proj_dim=128,      # her projeksiyon 128 dim → CatBoost'a 256 dim gider
        dropout=0.5,
        grl_lambda=1.0,
    ).to(device)

    print(f"Parametre   : {sum(p.numel() for p in model.parameters()):,}")
    print(f"CatBoost feature boyutu: {model.proj_dim * 2}")

    # ── CNN Egitim ────────────────────────────
    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        n_epochs=50,
        lr=1e-3,
        save_dir="checkpoints",
        alpha=0.5,   # target loss agirligi
        beta=0.5,    # triplet loss agirligi (person + target icin)
        gamma=0.3,   # session adversary agirligi
    )

    # ── En iyi modeli yukle ───────────────────
    ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Feature cikar ─────────────────────────
    x_train, y_train_person, y_train_target = extract_features_for_catboost(
        model, train_loader, device
    )
    x_test, y_test_person, y_test_target = extract_features_for_catboost(
        model, test_loader, device
    )
    x_diff, y_diff_person, y_diff_target = extract_features_for_catboost(
        model, different_loader, device
    )

    print(f"Feature boyutu: {x_train.shape}")

    '''# ── CatBoost ──────────────────────────────
    person_model = CatBoostClassifier(
        iterations=500, learning_rate=0.05,
        depth=4, l2_leaf_reg=10,
        verbose=100, random_seed=SEED
    )
    target_model = CatBoostClassifier(
        iterations=500, learning_rate=0.05,
        depth=4, l2_leaf_reg=10,
        verbose=100, random_seed=SEED
    )

    # Person modeli: tum train verisi
    person_model.fit(x_train, y_train_person)

    # Target modeli: SADECE genuine ornekler
    genuine_mask = (y_train_person == 1)
    target_model.fit(x_train[genuine_mask], y_train_target[genuine_mask])

    # ── Gorulmus session tahmini ──────────────
    print("\n=== GORULMUS SESSION ===")
    person_pred, target_pred = two_stage_predict(person_model, target_model, x_test)
    pipeline_score(y_test_person, y_test_target, person_pred, target_pred)

    # ── Gorulmemis session tahmini ────────────
    print("\n=== GORULMEMIS SESSION (8. session) ===")
    person_pred2, target_pred2 = two_stage_predict(person_model, target_model, x_diff)
    pipeline_score(y_diff_person, y_diff_target, person_pred2, target_pred2)
'''
    # ── Kaydet ────────────────────────────────
    np.save("train_features.npy", x_train)
    np.save("train_person.npy",   y_train_person)
    np.save("train_target.npy",   y_train_target)
    np.save("test_features.npy",  x_test)
    np.save("test_person.npy",    y_test_person)
    np.save("test_target.npy",    y_test_target)
    np.save("diff_features.npy",  x_diff)
    np.save("diff_person.npy",    y_diff_person)
    np.save("diff_target.npy",    y_diff_target)
    print("\nKaydedildi.")