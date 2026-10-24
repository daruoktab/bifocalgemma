import importlib.util
from collections.abc import Callable
from typing import NoReturn, cast

import pytest
import torch

from bifocalgemma.models.memory_efficient import attention_tile, tiled_attention


@pytest.mark.parametrize("width", [256, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_joint_gqa_output_and_gradients_at_native_head_widths(width, dtype):
    torch.manual_seed(19)
    q = torch.randn(1, 4, 7, width, dtype=dtype, requires_grad=True)
    k = torch.randn(1, 1, 12, width, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    enc_valid = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    dec_valid = torch.tensor([[0, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    from bifocalgemma.models.merged_attention import build_joint_visibility

    mask = build_joint_visibility(enc_valid, dec_valid, 3)
    # Match native Gemma eager's BF16 score/probability rounding. SDPA accumulates
    # differently and is not the numerical reference for this custom backward.
    expected = attention_tile(q, k, v, mask, width**-0.5, 0.0, True)
    actual = tiled_attention(
        q,
        k,
        v,
        chunk_size=3,
        scaling=width**-0.5,
        encoder_valid=enc_valid,
        decoder_valid=dec_valid,
        window=3,
    )
    tolerance = 0.02 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    expected_grads = torch.autograd.grad(
        expected.float().square().sum(), (q, k, v), retain_graph=True
    )
    grads = torch.autograd.grad(actual.float().square().sum(), (q, k, v))
    for left, right in zip(grads, expected_grads, strict=True):
        torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="CUDA Triton toolchain required for fused-kernel validation; target Molab preflight also checks this.",
)
@pytest.mark.parametrize("width", [256, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flex_joint_gqa_matches_reference_on_cuda(width, dtype):
    from bifocalgemma.models.flex_attention import fused_attention

    torch.manual_seed(19)
    q = torch.randn(1, 4, 17, width, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(1, 1, 30, width, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    enc = torch.ones(1, 13, device="cuda", dtype=torch.bool)
    dec = torch.ones(1, 17, device="cuda", dtype=torch.bool)
    reference = tiled_attention(
        q, k, v, chunk_size=8, scaling=width**-0.5, encoder_valid=enc, decoder_valid=dec, window=5
    )
    actual = fused_attention(
        q, k, v, scaling=width**-0.5, encoder_valid=enc, decoder_valid=dec, window=5
    )
    tolerance = 0.03 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
    reference_grads = torch.autograd.grad(
        reference.float().square().sum(), (q, k, v), retain_graph=True
    )
    try:
        gradients = torch.autograd.grad(actual.float().square().sum(), (q, k, v))
    except RuntimeError as error:
        if (
            width == 512
            and torch.cuda.get_device_capability() == (8, 6)
            and "out of resource" in str(error)
        ):
            # pytest's outcome decorator loses this signature in ty.
            cast(Callable[[str], NoReturn], pytest.xfail)(
                "Ampere sm86 shared-memory limit: hybrid backend tiles head512."
            )
        raise
    if dtype == torch.bfloat16:
        from bifocalgemma.models.merged_attention import build_joint_visibility

        # Native BF16 rounds scores and probabilities; fused attention accumulates
        # them differently. Check BOTH against FP32 rather than relaxing an
        # elementwise tolerance around near-zero BF16 gradients.
        fp_inputs = [x.detach().float().requires_grad_() for x in (q, k, v)]
        fp_output = attention_tile(
            fp_inputs[0],
            fp_inputs[1],
            fp_inputs[2],
            build_joint_visibility(enc, dec, 5),
            width**-0.5,
            0.0,
            True,
        )
        fp_grads = torch.autograd.grad(fp_output.square().sum(), fp_inputs)
        for candidate in (gradients, reference_grads):
            for grad, truth in zip(candidate, fp_grads, strict=True):
                relative_error = (grad.float() - truth).norm() / truth.norm().clamp_min(1e-8)
                assert torch.isfinite(grad).all() and float(relative_error) < 0.03
    else:
        for left, right in zip(gradients, reference_grads, strict=True):
            torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="CUDA Triton toolchain required.",
)
@pytest.mark.parametrize("bidirectional", [False, True])
def test_flex_structured_native_masks_across_sequence_lengths(bidirectional):
    from bifocalgemma.models.flex_attention import fused_attention

    torch.manual_seed(17)
    for length in (7, 35, 17):
        q = torch.randn(1, 4, length, 256, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 1, length, 256, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        valid = torch.ones(1, length, device="cuda", dtype=torch.bool)
        valid[:, :2] = False
        mask = {"valid": valid, "bidirectional": bidirectional}
        reference = tiled_attention(
            q, k, v, chunk_size=8, scaling=256**-0.5, attention_mask=mask, window=5
        )
        actual = fused_attention(q, k, v, scaling=256**-0.5, attention_mask=mask, window=5)
        torch.testing.assert_close(actual, reference, atol=0.03, rtol=0.03)
