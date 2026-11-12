"""World Models 的 V（VAE）與 M（MDN-LSTM / MDN-Transformer）。

V：64x64 畫面 ↔ 32 維潛在 z（Ha & Schmidhuber 2018 的架構）。
M：給過去的 (z, a)，用混合高斯 (MDN) 預測下一個 z 的分布。兩種骨幹可切換：
   - LSTM：原論文，每步 O(1)，狀態是 h_t
   - Transformer：因果注意力，context 視窗內每一幀都能直接回看
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Z_DIM, A_DIM = 32, 3


class VAE(nn.Module):
    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(), nn.Conv2d(128, 256, 4, 2), nn.ReLU(), nn.Flatten())
        self.mu, self.logvar = nn.Linear(1024, z_dim), nn.Linear(1024, z_dim)
        self.fc = nn.Linear(z_dim, 1024)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, 2), nn.ReLU(), nn.ConvTranspose2d(128, 64, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2), nn.ReLU(), nn.ConvTranspose2d(32, 3, 6, 2), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h).clamp(-8, 8)

    def decode(self, z):
        return self.dec(self.fc(z)[:, :, None, None])

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.decode(z), mu, logvar


class MDNHead(nn.Module):
    """每個 z 維度獨立一個 K 成分的混合高斯。"""
    def __init__(self, hidden, z_dim=Z_DIM, k=5):
        super().__init__()
        self.z_dim, self.k = z_dim, k
        self.out = nn.Linear(hidden, z_dim * k * 3)

    def forward(self, h):
        o = self.out(h).view(*h.shape[:-1], self.z_dim, self.k, 3)
        return F.log_softmax(o[..., 0], -1), o[..., 1], o[..., 2].clamp(-7, 3)   # logpi, mu, logsigma

    @staticmethod
    def nll(params, z):
        logpi, mu, logsig = params
        z = z.unsqueeze(-1)
        logp = -0.5 * ((z - mu) / logsig.exp()) ** 2 - logsig - 0.5 * math.log(2 * math.pi)
        return -torch.logsumexp(logpi + logp, -1).mean()

    @staticmethod
    def sample(params, temperature=1.0):
        logpi, mu, logsig = params
        if temperature == 0:      # 取期望值（確定性預測）
            return (logpi.exp() * mu).sum(-1)
        idx = torch.distributions.Categorical(logits=logpi / temperature).sample()
        m = mu.gather(-1, idx[..., None])[..., 0]
        s = logsig.exp().gather(-1, idx[..., None])[..., 0]
        return m + torch.randn_like(m) * s * math.sqrt(temperature)


class MDNLSTM(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.rnn = nn.LSTM(Z_DIM + A_DIM, hidden, batch_first=True)
        self.head = MDNHead(hidden)

    def forward(self, z, a, state=None):
        h, state = self.rnn(torch.cat([z, a], -1), state)
        return self.head(h), h, state


class MDNTransformer(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, context=32):
        super().__init__()
        self.context = context
        self.inp = nn.Linear(Z_DIM + A_DIM, d)
        self.pos = nn.Parameter(torch.zeros(context, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.head = MDNHead(d)

    def forward(self, z, a, state=None):
        x = torch.cat([z, a], -1)[:, -self.context:]
        T = x.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        h = self.tf(self.inp(x) + self.pos[:T], mask=mask)
        return self.head(h), h, None


def make_m(kind):
    return {"lstm": MDNLSTM, "transformer": MDNTransformer}[kind]()
