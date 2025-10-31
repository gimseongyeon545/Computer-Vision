# transformer example code
# multimodal transformer (cnn encoder (image) + mlp encoder (q7) + decoder)


from typing import Dict, Optional
import math
import torch
import torch.nn as nn

# -----------------------------
# Positional Encoding (sin/cos)
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)  # [L, D]
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        return x + self.pe[:, : x.size(1)]


# -----------------------------
# Encoders
# -----------------------------
class ConvEncoder(nn.Module):
    """Lightweight CNN → vector. Input: [N, C, H, W] → [N, d_out]."""

    def __init__(self, in_ch: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ReLU(),  # H/2, W/2
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),      # H/4, W/4
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),     # H/8, W/8
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.proj(h)


class MLPEncoder(nn.Module):
    """Vector encoder. Input: [B, K, D] → [B, K, d_out]."""

    def __init__(self, in_dim: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Multi-modal Transformer (generalized)
# -----------------------------
class MultiModalTransformer(nn.Module):
    """
    Generalized multi-modal temporal encoder + learned-query decoder.

    Inputs (dict of tensors):
      - For each modality configured as kind="image": tensor [B, K, C, H, W]
      - For each modality configured as kind="vector": tensor [B, K, D]

    Output:
      - pred: [B, T, out_dim]
    """

    def __init__(
        self,
        modalities: Dict[str, Dict],  # e.g., {"rgb":{"kind":"image","in_ch":3,"dim":128}, "q":{"kind":"vector","in_dim":7,"dim":256}}
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pred_horizon: int = 16,
        out_dim: int = 7,
        use_tanh_out: bool = True,
        max_len: int = 4096,
    ):
        super().__init__()
        self.modalities = modalities
        self.d_model = d_model
        self.pred_horizon = pred_horizon
        self.out_dim = out_dim
        self.use_tanh_out = use_tanh_out

        encoders = {}
        fuse_in = 0
        for name, spec in modalities.items():
            kind = spec["kind"]
            dim = spec.get("dim", d_model)
            if kind == "image":
                encoders[name] = ConvEncoder(in_ch=spec["in_ch"], d_out=dim)
            elif kind == "vector":
                encoders[name] = MLPEncoder(in_dim=spec["in_dim"], d_out=dim)
            else:
                raise ValueError(f"Unknown modality kind: {kind}")
            fuse_in += dim
        self.encoders = nn.ModuleDict(encoders)

        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.future_queries = nn.Parameter(torch.zeros(pred_horizon, d_model))
        nn.init.trunc_normal_(self.future_queries, std=0.02)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim),
        )

    @staticmethod
    def _encode_image(mod: ConvEncoder, x: torch.Tensor) -> torch.Tensor:
        """x: [B, K, C, H, W] → [B, K, D]"""
        B, K = x.shape[:2]
        x_bk = x.contiguous().view(B * K, *x.shape[2:])  # [B*K, C, H, W]
        feat = mod(x_bk)  # [B*K, D]
        return feat.view(B, K, -1)

    @staticmethod
    def _encode_vector(mod: MLPEncoder, x: torch.Tensor) -> torch.Tensor:
        """x: [B, K, D_in] → [B, K, D]"""
        return mod(x)

    def _encode_concat(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode per modality and concat on feature dim → [B, K, d_model]."""
        feats = []
        B, K = None, None
        for name, spec in self.modalities.items():
            x = inputs[name]
            if B is None:
                if spec["kind"] == "image":
                    B, K = x.size(0), x.size(1)
                else:
                    B, K = x.size(0), x.size(1)
            enc = self.encoders[name]
            if spec["kind"] == "image":
                f = self._encode_image(enc, x)  # [B, K, dim]
            else:  # vector
                f = self._encode_vector(enc, x)  # [B, K, dim]
            feats.append(f)
        fused = torch.cat(feats, dim=-1)  # [B, K, sum(dim_i)]
        tokens = self.fuse(fused)  # [B, K, d_model]
        return tokens

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],  # keys must match self.modalities
        pred_horizon: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs: dict mapping modality name → tensor.
            pred_horizon: optional int to override default T.
        Returns:
            [B, T, out_dim]
        """
        T = pred_horizon if pred_horizon is not None else self.pred_horizon
        assert 1 <= T <= self.pred_horizon

        # Encode observation sequence
        tokens = self._encode_concat(inputs)              # [B, K, D]
        tokens = self.pos_enc(tokens)
        memory = self.encoder(tokens)                     # [B, K, D]

        # Learned future queries
        B = tokens.size(0)
        qrys = self.future_queries[:T, :].unsqueeze(0).expand(B, T, self.d_model)
        qrys = self.pos_enc(qrys)

        # Decode T steps and predict
        h = self.decoder(tgt=qrys, memory=memory)         # [B, T, D]
        out = self.head(h)                                # [B, T, out_dim]
        if self.use_tanh_out:
            out = torch.tanh(out)
        return out


# -----------------------------
# Minimal demo
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define modalities (similar to your original)
    mods = {
        "rgb":   {"kind": "image",  "in_ch": 3, "dim": 128},
        "depth": {"kind": "image",  "in_ch": 1, "dim": 128},
        "q":     {"kind": "vector", "in_dim": 7, "dim": 256},
    }

    model = MultiModalTransformer(
        modalities=mods,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        pred_horizon=8,
        out_dim=7,
        use_tanh_out=True,
    ).to(device)

    B, K = 2, 6
    H, W = 96, 96
    dummy = {
        "rgb":   torch.rand(B, K, 3, H, W, device=device),
        "depth": torch.rand(B, K, 1, H, W, device=device),
        "q":     torch.rand(B, K, 7, device=device) * 2 - 1,
    }

    y = model(dummy, pred_horizon=8)
    print("Output:", y.shape)  # [B, T, 7]
