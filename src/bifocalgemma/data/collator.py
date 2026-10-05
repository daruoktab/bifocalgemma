"""D02: left-padded, patch-only teacher forcing. See docs/DATA_PIPELINE.md."""

import torch

from bifocalgemma.contracts import Seq2SeqBatch, TokenizedRepairExample


class RepairCollator:
    def __init__(self, pad_token_id: int = 0, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, examples: list[TokenizedRepairExample]) -> Seq2SeqBatch:
        if not examples:
            raise ValueError("Cannot collate an empty batch.")
        encoder_presence = [bool(example.encoder_ids) for example in examples]
        if any(encoder_presence) and not all(encoder_presence):
            raise ValueError("Do not mix decoder-only and encoder-decoder examples in one batch.")
        for example in examples:
            if not example.decoder_prefix_ids or not example.target_ids:
                raise ValueError("Decoder prefix and target must each contain tokens.")
            if min((*example.encoder_ids, *example.decoder_prefix_ids, *example.target_ids)) < 0:
                raise ValueError("Input token IDs must be nonnegative.")
        enc_len = max(len(example.encoder_ids) for example in examples)
        dec_len = max(
            len(example.decoder_prefix_ids) + len(example.target_ids) - 1 for example in examples
        )
        size = len(examples)
        encoder = torch.full((size, enc_len), self.pad_token_id, dtype=torch.long)
        decoder = torch.full((size, dec_len), self.pad_token_id, dtype=torch.long)
        enc_mask = torch.zeros_like(encoder, dtype=torch.bool)
        dec_mask = torch.zeros_like(decoder, dtype=torch.bool)
        labels = torch.full_like(decoder, self.ignore_index)
        for row, example in enumerate(examples):
            encoder[row, -len(example.encoder_ids) :] = torch.tensor(example.encoder_ids)
            enc_mask[row, -len(example.encoder_ids) :] = True
            combined = example.decoder_prefix_ids + example.target_ids
            inputs = combined[:-1]
            targets = list(combined[1:])
            targets[: len(example.decoder_prefix_ids) - 1] = [self.ignore_index] * (
                len(example.decoder_prefix_ids) - 1
            )
            decoder[row, -len(inputs) :] = torch.tensor(inputs)
            dec_mask[row, -len(inputs) :] = True
            labels[row, -len(targets) :] = torch.tensor(targets)
        return Seq2SeqBatch(encoder, enc_mask, decoder, dec_mask, labels)
