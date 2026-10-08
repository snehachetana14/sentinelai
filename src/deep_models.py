"""
deep_models.py - the two deep-learning models (PyTorch).

1. DNNClassifier  - a deep neural network that names the attack type (supervised).
2. Autoencoder    - learns to rebuild NORMAL traffic; a big rebuild error = anomaly.

Both have the same simple interface: .fit(...), .save(path), .load(path)
so the rest of the project never has to touch PyTorch directly.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(X, y=None, batch_size=512, shuffle=False):
    tensors = [torch.as_tensor(X, dtype=torch.float32)]
    if y is not None:
        tensors.append(torch.as_tensor(y, dtype=torch.long))
    # while training (shuffle=True) we drop a tiny leftover batch: BatchNorm needs >1 row
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      drop_last=shuffle and len(X) > batch_size)


# =====================================================================  DNN classifier
class _DNN(nn.Module):
    """input -> 128 -> 64 -> 5 classes.  Dropout randomly switches neurons off
    during training so the network can't just memorise the data."""

    def __init__(self, n_in, n_out, hidden=(128, 64), dropout=0.3):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)          # raw scores ("logits"); softmax turns them into probabilities


class DNNClassifier:
    def __init__(self, n_features, n_classes, hidden=(128, 64), dropout=0.3,
                 lr=1e-3, epochs=15, batch_size=512, seed=42):
        self.config = dict(n_features=n_features, n_classes=n_classes, hidden=list(hidden),
                           dropout=dropout, lr=lr, epochs=epochs, batch_size=batch_size, seed=seed)
        torch.manual_seed(seed)
        self.model = _DNN(n_features, n_classes, hidden, dropout).to(DEVICE)

    def fit(self, X, y, X_val=None, y_val=None):
        c = self.config
        opt = torch.optim.Adam(self.model.parameters(), lr=c["lr"])
        loss_fn = nn.CrossEntropyLoss()
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(1, c["epochs"] + 1):
            self.model.train()
            total, seen = 0.0, 0
            for xb, yb in _loader(X, y, c["batch_size"], shuffle=True):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
                seen += len(xb)
            history["train_loss"].append(total / seen)
            msg = f"    epoch {epoch:2d}/{c['epochs']}  train loss {history['train_loss'][-1]:.4f}"
            if X_val is not None:
                p = self.predict_proba(X_val)
                val_loss = -np.mean(np.log(p[np.arange(len(y_val)), y_val] + 1e-9))
                history["val_loss"].append(float(val_loss))
                msg += f"  val loss {val_loss:.4f}"
            print(msg)
        return history

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        out = [torch.softmax(self.model(xb.to(DEVICE)), dim=1).cpu().numpy()
               for (xb,) in _loader(X, batch_size=4096)]
        return np.concatenate(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path):
        torch.save({"config": self.config, "state": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        c = ckpt["config"]
        obj = cls(c["n_features"], c["n_classes"], tuple(c["hidden"]), c["dropout"],
                  c["lr"], c["epochs"], c["batch_size"], c["seed"])
        obj.model.load_state_dict(ckpt["state"])
        obj.model.eval()
        return obj


# =====================================================================  Autoencoder
class _AE(nn.Module):
    """ENCODER squeezes a row:  n -> 64 -> 32 -> 8   (the 'bottleneck')
       DECODER rebuilds it:     8 -> 32 -> 64 -> n
    Sigmoid at the end because every input feature is scaled to 0..1."""

    def __init__(self, n_in, bottleneck=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, n_in), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Autoencoder:
    def __init__(self, n_features, bottleneck=8, lr=1e-3, epochs=30, batch_size=512, seed=42):
        self.config = dict(n_features=n_features, bottleneck=bottleneck, lr=lr,
                           epochs=epochs, batch_size=batch_size, seed=seed)
        torch.manual_seed(seed)
        self.model = _AE(n_features, bottleneck).to(DEVICE)

    def fit(self, X_normal, X_val_normal=None):
        """Train ONLY on normal traffic. Target = the input itself."""
        c = self.config
        opt = torch.optim.Adam(self.model.parameters(), lr=c["lr"])
        loss_fn = nn.MSELoss()
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(1, c["epochs"] + 1):
            self.model.train()
            total, seen = 0.0, 0
            for (xb,) in _loader(X_normal, batch_size=c["batch_size"], shuffle=True):
                xb = xb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), xb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
                seen += len(xb)
            history["train_loss"].append(total / seen)
            msg = f"    epoch {epoch:2d}/{c['epochs']}  train MSE {history['train_loss'][-1]:.6f}"
            if X_val_normal is not None:
                history["val_loss"].append(float(self.reconstruction_error(X_val_normal).mean()))
                msg += f"  val MSE {history['val_loss'][-1]:.6f}"
            print(msg)
        return history

    @torch.no_grad()
    def reconstruction_error(self, X):
        """Mean squared difference between each row and its rebuilt version."""
        self.model.eval()
        out = []
        for (xb,) in _loader(X, batch_size=4096):
            xb = xb.to(DEVICE)
            out.append(((self.model(xb) - xb) ** 2).mean(dim=1).cpu().numpy())
        return np.concatenate(out)

    def save(self, path):
        torch.save({"config": self.config, "state": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        c = ckpt["config"]
        obj = cls(c["n_features"], c["bottleneck"], c["lr"], c["epochs"], c["batch_size"], c["seed"])
        obj.model.load_state_dict(ckpt["state"])
        obj.model.eval()
        return obj
