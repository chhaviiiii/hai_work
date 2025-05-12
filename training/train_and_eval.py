#!/usr/bin/env python3
import os, glob, pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR      = "/ubc/cs/home/c/cnayyar/hai_work"
PKL_DIR       = os.path.join(BASE_DIR, "output")
CKPT_PATH     = os.path.join(BASE_DIR, "trained", "vtnet_best.pth")
BATCH_SIZE    = 16
EPOCHS        = 5
LR            = 1e-3
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# inline labels: map your subj IDs (unknown_x_y → ctrl_x_y) to 0/1 here
VERBALWM_LABELS = {
    "ctrl_1_20": 1, "ctrl_1_27": 1, "ctrl_1_30": 1,
    # … complete for all your subjects …
}
# ────────────────────────────────────────────────────────────────────────────────

class VTNetDataset(Dataset):
    def __init__(self, pkl_list):
        self.files = pkl_list
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        p = self.files[idx]
        base = os.path.basename(p).rsplit(".",1)[0]
        scid = base.replace("unknown","ctrl")
        df = pd.read_pickle(p)
        seq = torch.from_numpy(df.values.astype(float)).float()
        lab = VERBALWM_LABELS.get(scid, 0)
        return seq, torch.tensor(lab, dtype=torch.float32)

def collate_fn(batch):
    seqs, labs = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True)
    return padded, torch.stack(labs)

class SimpleGRU(nn.Module):
    def __init__(self, feat_dim, hidden=64):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)
    def forward(self, x):
        _, h = self.gru(x)
        h = h.squeeze(0)
        return self.fc(h).squeeze(1)

def main():
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

    all_pkls = sorted(glob.glob(os.path.join(PKL_DIR, "*.pkl")))
    train_p, val_p = train_test_split(all_pkls, test_size=0.2, random_state=42)

    train_dl = DataLoader(VTNetDataset(train_p), batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)
    val_dl   = DataLoader(VTNetDataset(val_p),   batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn)

    # infer feature Dim
    feat_dim = pd.read_pickle(all_pkls[0]).shape[1]
    model = SimpleGRU(feat_dim).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(1, EPOCHS+1):
        model.train()
        tot_loss = 0.0
        for X, y in train_dl:
            X,y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()*X.size(0)
        print(f"[{epoch}] Train Loss: {tot_loss/len(train_dl.dataset):.4f}")

        # eval
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for X,y in val_dl:
                X,y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                preds.append(out.sigmoid().cpu())
                truths.append(y.cpu())
        p = torch.cat(preds).numpy()
        t = torch.cat(truths).numpy()
        auc = roc_auc_score(t, p)
        acc = accuracy_score(t, (p>0.5).astype(int))
        print(f"      Val AUC: {auc:.4f}  Acc: {acc:.4f}")
        # save best
        if auc > best_auc:
            best_auc=auc
            torch.save(model.state_dict(), CKPT_PATH)
    print("Done. Best Val AUC:", best_auc, "→ saved to", CKPT_PATH)

if __name__=="__main__":
    main()
