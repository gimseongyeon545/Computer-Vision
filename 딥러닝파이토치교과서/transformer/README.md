# MultiModalTransformer
- `inputs: Dict[str, torch.Tensor]`: 빈 dict
- `T`: T_p (예측할 길이 `pred_horizon`)
## `__init__` => **받은 인자를 통해서 각 레이어의 틀 만들기**
- `self.modalities` 인자로 받은 mods 에서 kind key 값 뽑고, dim 의 경우 없으면 .get 으로 d_model 로 value 지정
  ```
  mods = {
        "rgb":   {"kind": "image",  "in_ch": 3, "dim": 128},
        "depth": {"kind": "image",  "in_ch": 1, "dim": 128},
        "q":     {"kind": "vector", "in_dim": 7, "dim": 256},
      }
  ```
 - `kind` 에 따라서 `ConvEncoder`, `MLPEncoder` 결과를 `encoders` 라는 dict 에 `encoders[name] = ` 을 통해 받은 인자를 통해 만들어진 encoder layer 를 넣음
 - ModuleDict 로 만들어서 `self.encoders`
   ```
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
   ```
- `self.fuse`, `self.pos_enc`, `self.head`
- `nn.TransformerEncoderLayer` / `nn.TransformerDecoderLayer` / `nn.TransformerEncoder` / `nn.TransformerDecoder` 틀을 만들기
- `pred_horizon` (예측할 길이 T_p) * `d_model` (토큰별 임베딩 길이) 크기의 tensor 를 zero 로 채운 후, std = 0.02, 평균 = 0 인 `trunc_normal_` (절단정규분포) 로 초기화 하기

</br>

## `forward`
[1] encoding
1) `_encode_concat(inputs)`
  - `self.modalities.items()`
    ```
    mods = {
      "rgb":   {"kind": "image",  "in_ch": 3, "dim": 128},
      "depth": {"kind": "image",  "in_ch": 1, "dim": 128},
      "q":     {"kind": "vector", "in_dim": 7, "dim": 256},
    }
    ```
  - `enc = self.encoders[name]`
    
2) positional encoding
3) Encoder
  - `nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)`
[2] Decoding
  - positional encoding
  - Decoder
    - `nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)`
  - head
  - tanh
