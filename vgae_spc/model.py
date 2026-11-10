"""Variational Graph Autoencoder (Kipf & Welling, 2016) en PyTorch puro.

Además del decoder de estructura (producto interno), incluye un decoder de
atributos que reconstruye la ventana temporal de cada sensor: con un grafo
fijo, el error de atributos es la señal útil para detectar anomalías.

Variante temporal (temporal="tcn"): en lugar de tratar la ventana de W horas como
un vector plano, cada sensor pasa por convoluciones 1D dilatadas sobre el eje del
tiempo (estilo TCN / STGCN) antes de la GCN, y el decoder aplica el bloque espejo.
La hora del día (sen, cos) entra como canales adicionales en ambos bloques.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(A):
    """D^-1/2 (A + I) D^-1/2"""
    A = A + torch.eye(A.shape[0], device=A.device)
    d = A.sum(1).pow(-0.5)
    return d[:, None] * A * d[None, :]


class GCNLayer(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        self.lin = nn.Linear(fin, fout)

    def forward(self, x, a_hat):          # x: (B, N, F)
        return torch.einsum("ij,bjf->bif", a_hat, self.lin(x))


TOD_CH = 2  # canales de hora del día: sen y cos


class TemporalBlock(nn.Module):
    """Convolución 1D dilatada con conexión residual; conserva la longitud de la ventana."""

    def __init__(self, cin, cout, dilation, k=3):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, padding=dilation * (k - 1) // 2, dilation=dilation)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):                 # x: (B·N, C, W)
        return F.relu(self.conv(x) + self.res(x))


class TCN(nn.Module):
    """Pila de bloques con dilataciones 1, 2, 4: campo receptivo de 15 pasos."""

    def __init__(self, cin, ch, dilations=(1, 2, 4)):
        super().__init__()
        self.blocks = nn.Sequential(*[TemporalBlock(cin if i == 0 else ch, ch, d) for i, d in enumerate(dilations)])

    def forward(self, x):
        return self.blocks(x)


def _with_tod(x, tod):
    """x: (B, N, W) -> (B·N, 1 + TOD_CH, W), repitiendo la hora del día en cada nodo."""
    B, N, W = x.shape
    t = tod.transpose(1, 2).unsqueeze(1).expand(B, N, TOD_CH, W)
    return torch.cat([x.unsqueeze(2), t], 2).reshape(B * N, 1 + TOD_CH, W)


class VGAE(nn.Module):
    def __init__(self, n_nodes, window, hidden=64, latent=16, emb=8, temporal="none", tcn_ch=16):
        super().__init__()
        self.temporal, self.window, self.tcn_ch = temporal, window, tcn_ch
        self.node_emb = nn.Parameter(0.1 * torch.randn(n_nodes, emb))
        fin = window * tcn_ch if temporal == "tcn" else window
        if temporal == "tcn":
            self.enc_tcn = TCN(1 + TOD_CH, tcn_ch)
            self.dec_tcn = TCN(tcn_ch + TOD_CH, tcn_ch, dilations=(4, 2, 1))
            self.dec_conv_out = nn.Conv1d(tcn_ch, 1, 1)
        self.gc1 = GCNLayer(fin + emb, hidden)
        self.gc_mu = GCNLayer(hidden, latent)
        self.gc_logvar = GCNLayer(hidden, latent)
        self.dec_gc = GCNLayer(latent + emb, hidden)
        self.dec_out = nn.Linear(hidden, fin)

    def encode(self, x, a_hat, tod=None):
        B, N, W = x.shape
        if self.temporal == "tcn":
            x = self.enc_tcn(_with_tod(x, tod)).reshape(B, N, self.tcn_ch * W)
        e = self.node_emb.expand(B, -1, -1)
        h = F.relu(self.gc1(torch.cat([x, e], -1), a_hat))
        return self.gc_mu(h, a_hat), self.gc_logvar(h, a_hat).clamp(-10, 10)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def decode_structure(self, z):        # logits de A_ij = z_i · z_j
        return z @ z.transpose(1, 2)

    def decode_features(self, z, a_hat, tod=None):
        B, N, _ = z.shape
        e = self.node_emb.expand(B, -1, -1)
        h = self.dec_out(F.relu(self.dec_gc(torch.cat([z, e], -1), a_hat)))
        if self.temporal != "tcn":
            return h
        W = self.window
        h = h.reshape(B * N, self.tcn_ch, W)
        t = tod.transpose(1, 2).unsqueeze(1).expand(B, N, TOD_CH, W).reshape(B * N, TOD_CH, W)
        h = self.dec_tcn(torch.cat([h, t], 1))
        return self.dec_conv_out(h).reshape(B, N, W)

    def forward(self, x, a_hat, tod=None):
        mu, logvar = self.encode(x, a_hat, tod)
        z = self.reparameterize(mu, logvar)
        return self.decode_features(z, a_hat, tod), self.decode_structure(z), mu, logvar


def vgae_loss(x, x_hat, adj_logits, A, mu, logvar, alpha=0.1, beta=1e-3):
    n = A.shape[0]
    rec_x = F.mse_loss(x_hat, x)
    pos_weight = (n * n - A.sum()) / A.sum()
    target = A.expand_as(adj_logits)
    rec_a = F.binary_cross_entropy_with_logits(adj_logits, target, pos_weight=pos_weight)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1))
    # los componentes se devuelven como tensores: convertirlos con .item() en cada paso
    # obligaría a la CPU a esperar a la GPU y la dejaría ociosa entre pasos
    return rec_x + alpha * rec_a + beta * kl, (rec_x.detach(), rec_a.detach(), kl.detach())
