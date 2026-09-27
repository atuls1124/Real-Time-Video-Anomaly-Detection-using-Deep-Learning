"""
Convolutional AutoEncoder for unsupervised anomaly detection (Avenue normal only).
Lightweight for GTX 1650 4GB: input 128x128 grayscale, latent 256.
Train: python -m src.anomaly.cae --train --epochs 15
Infer: model.predict(frame) -> mse score > thresh => anomaly
"""
import torch, torch.nn as nn
from pathlib import Path
import cv2, numpy as np

class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,3,2,1), nn.ReLU(), # 64
            nn.Conv2d(32,64,3,2,1), nn.ReLU(), #32
            nn.Conv2d(64,128,3,2,1), nn.ReLU(), #16
            nn.Conv2d(128,256,3,2,1), nn.ReLU(), #8
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(), #16
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(), #32
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(), #64
            nn.ConvTranspose2d(32,1,4,2,1), nn.Sigmoid(), #128
        )
    def forward(self,x):
        z=self.encoder(x)
        return self.decoder(z)

def preprocess(frame, size=128):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(size,size))
    t = torch.from_numpy(gray).float()/255.0
    return t.unsqueeze(0).unsqueeze(0) # 1x1xHxW

class AnomalyCAE:
    def __init__(self, ckpt=None, device="cuda" if torch.cuda.is_available() else "cpu", thresh=None):
        self.device=device
        self.model=ConvAE().to(device)
        self.thresh=thresh
        if ckpt and Path(ckpt).exists():
            self.model.load_state_dict(torch.load(ckpt, map_location=device))
            self.model.eval()
            print(f"Loaded CAE {ckpt}")
        self.train_thresh=None

    def train(self, train_frames, epochs=15, lr=1e-3, save_path="models/cae_avenue.pt"):
        self.model.train()
        opt=torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn=nn.MSELoss()
        for epoch in range(epochs):
            total=0
            for f in train_frames:
                x=preprocess(f).to(self.device)
                opt.zero_grad()
                out=self.model(x)
                loss=loss_fn(out,x)
                loss.backward(); opt.step()
                total+=loss.item()
            print(f"CAE epoch {epoch+1}/{epochs} loss {total/len(train_frames):.5f}")
        # calibrate thresh on train set
        mses=[]
        self.model.eval()
        with torch.no_grad():
            for f in train_frames[:200]:
                x=preprocess(f).to(self.device)
                out=self.model(x)
                mses.append(nn.functional.mse_loss(out,x).item())
        self.train_thresh=float(np.percentile(mses,95))
        print(f"CAE thresh 95pct {self.train_thresh:.5f}")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        self.thresh=self.train_thresh
        return self.train_thresh

    def score(self, frame):
        self.model.eval()
        with torch.no_grad():
            x=preprocess(frame).to(self.device)
            out=self.model(x)
            mse = nn.functional.mse_loss(out,x).item()
        is_anom = mse > self.thresh if self.thresh else False
        return mse, is_anom
