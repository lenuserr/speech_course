from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SubDecoder(nn.Module):
    def __init__(self, d_model, n_codes, n_codebooks):
        super().__init__()
        self.n_codes, self.n_codebooks = n_codes, n_codebooks
        self.d_model = d_model

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
        self.decoder = nn.TransformerEncoder(encoder_layer=layer, num_layers=2)
        self.out_proj = nn.Linear(in_features=d_model, out_features=n_codes)

        self.codes_embedder = nn.Embedding(num_embeddings=n_codes, embedding_dim=d_model)
        self.positional_encoding = nn.Embedding(num_embeddings=n_codebooks, embedding_dim=d_model)

        self.codes_positional_encoding = nn.Embedding(num_embeddings=n_codebooks + 1, embedding_dim=d_model)

    def forward(
        self,
        emb_sequence, 
        codes_sequence, 
    ):
        device = emb_sequence.device

        codes_embeddings = self.codes_embedder(codes_sequence)
        B, L, N, d = codes_embeddings.shape

        emb_sequence = emb_sequence.view(B * L, 1, d)
        codes_embeddings = codes_embeddings.view(B * L, N, d)

        codes_range = torch.arange(N + 1, device=device).unsqueeze(dim=0)
        codes_PE = self.codes_positional_encoding(codes_range)

        src = torch.cat([emb_sequence, codes_embeddings], dim=1)
        src = src + codes_PE

        embeddings = self.decoder(
            src=src,
            mask=nn.Transformer.generate_square_subsequent_mask(N + 1, device=device)
        )
        embeddings = self.out_proj(embeddings)
        return embeddings.view(B, L, N + 1, self.n_codes)

    @torch.no_grad()
    def autoregressive_sampling(
        self,
        embedding,
        sampling_fn: Callable = lambda x: x.argmax(dim=-1),
    ):
        B, d = embedding.shape
        device = embedding.device
        assert B == 1, "Batch size should be 1"

        emb_sequence = embedding.unsqueeze(dim=1) 

        codes_sequence = torch.zeros(
            B, 1, 0, dtype=torch.long, device=device,
        )

        for _ in range(self.n_codebooks):
            logits = self.forward(
                emb_sequence=emb_sequence,
                codes_sequence=codes_sequence,
            )
            next_logits = logits[:, :, -1, :]             
            next_token = sampling_fn(next_logits)           
            codes_sequence = torch.cat(
                [codes_sequence, next_token.unsqueeze(-1)], dim=-1,
            )                                               

        return codes_sequence 


class EncoderDecoder(nn.Module):
    def __init__(self, d_model, n_phonemes, n_codes, n_codebooks):
        super().__init__()

        self.phoneme_embedding = nn.Embedding(num_embeddings=n_phonemes, embedding_dim=d_model)

        assert d_model % n_codebooks == 0, f"{d_model=} {n_codebooks=}"
        self.codes_embedding = nn.ModuleList([
            nn.Embedding(num_embeddings=n_codes, embedding_dim=d_model // n_codebooks)
            for _ in range(n_codebooks)
        ])

        self.phones_positional_encoding = nn.Embedding(num_embeddings=1000, embedding_dim=d_model)
        self.n_pos_embs = 2300
        self.codes_positional_encoding = nn.Embedding(num_embeddings=self.n_pos_embs, embedding_dim=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=8,
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=8,
        )

    def forward(
        self,
        phones, 
        phones_mask, 
        codes, 
        codes_mask, 
        speaker_embs, 
    ):
        device=phones.device

        phone_embs = self.phoneme_embedding(phones)
        phone_embs = torch.cat((speaker_embs.unsqueeze(dim=1), phone_embs), dim=1)
        mask_complement = torch.ones(phones.shape[0], 1, device=device, dtype=torch.bool)
        phones_mask = torch.cat((mask_complement, phones_mask), dim=1)

        phones_range = torch.arange(phones.shape[1] + 1, device=device).unsqueeze(dim=0)
        phones_PE = self.phones_positional_encoding(phones_range)
        phones_inp = phone_embs + phones_PE

        codes_embs = [emb_layer(codes[:, :, idx]) for idx, emb_layer in enumerate(self.codes_embedding)]
        codes_embs = torch.cat(codes_embs, dim=2)

        codes_range = torch.arange(codes.shape[1], device=device).unsqueeze(dim=0)
        codes_range = torch.clamp(codes_range, 0, self.n_pos_embs - 1) 
        codes_PE = self.codes_positional_encoding(codes_range)
        codes_inp = codes_embs + codes_PE

        phonemes_encoded = self.encoder(
            src=phones_inp,
            mask=None,
            src_key_padding_mask=~phones_mask
        )
        embeddings = self.decoder(
            tgt=codes_inp,
            memory=phonemes_encoded,
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(codes.shape[1], device=device).bool(),
            memory_mask=None,
            tgt_key_padding_mask=~codes_mask,
            memory_key_padding_mask=~phones_mask,
        )

        return embeddings


class TTSTransformer(nn.Module):
    def __init__(self, n_phonemes: int, n_codes: int, n_codebooks: int):
        super().__init__()
        d_model = 512 + 256

        self.speaker_linear = nn.Linear(512, d_model)

        self.encoder_decoder = EncoderDecoder(
            d_model=d_model,
            n_phonemes=n_phonemes,
            n_codes=n_codes,
            n_codebooks=n_codebooks,
        )

        self.subdecoder = SubDecoder(
            d_model=d_model,
            n_codes=n_codes,
            n_codebooks=n_codebooks,
        )

    def forward(
        self,
        phones, 
        phones_mask, 
        codes, 
        codes_mask, 
        speaker_embs, 
    ):
        speaker_embs = self.speaker_linear(speaker_embs)

        embeddings = self.encoder_decoder(
            phones=phones,
            phones_mask=phones_mask,
            codes=codes,
            codes_mask=codes_mask,
            speaker_embs=speaker_embs,
        )

        prediction = self.subdecoder(embeddings, codes)[:, :, :-1, :]

        return prediction

    @torch.no_grad()
    def autoregressive_sampling(
        self,
        phones, 
        speaker_embs, 
        max_size: int = 1000,
        start_token: int = 161,
        end_token: int = 160,
        sampling_fn: Callable = lambda x: x.argmax(dim=-1),
    ):
        batch_size = phones.shape[0]
        assert batch_size == 1, "Batch size must be 1"
        device = phones.device

        n_codebooks = self.subdecoder.n_codebooks

        speaker_embs_proj = self.speaker_linear(speaker_embs)  

        phones_mask = torch.ones_like(phones, dtype=torch.bool)

        codes = torch.full(
            (batch_size, 1, n_codebooks),
            fill_value=start_token,
            dtype=torch.long,
            device=device,
        )

        for _ in tqdm(range(max_size), desc="autoregressive sampling", leave=False):
            codes_mask = torch.ones(codes.shape[:2], dtype=torch.bool, device=device)

            embeddings = self.encoder_decoder(
                phones=phones,
                phones_mask=phones_mask,
                codes=codes,
                codes_mask=codes_mask,
                speaker_embs=speaker_embs_proj,
            )                                         

            last_embedding = embeddings[:, -1, :]      

            next_codes = self.subdecoder.autoregressive_sampling(
                embedding=last_embedding,
                sampling_fn=sampling_fn,
            )                                         

            codes = torch.cat([codes, next_codes], dim=1)

            if (next_codes == end_token).any():
                break

        return codes[:, 1:, :]
