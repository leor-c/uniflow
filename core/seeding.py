"""Named RNG streams, so runs are reproducible *and* comparable.

A single ``torch.manual_seed(seed)`` at the top of a run makes that run
reproducible, but it does not make two runs comparable. Every consumer --
weight init, the data sampler, the training noise, the evaluation draws --
then shares one stream, so what any of them sees depends on how many numbers
everything before it consumed. Two arms with different architectures diverge on
their first parameter tensor and never realign: with the same seed the toy arms
were trained and evaluated on *different* data.

Deriving one stream per role from the same seed fixes that. Anything drawn from
``(seed, stream)`` is a function of that pair alone, so shared roles -- the
data, the evaluation set -- are identical across arms, while each arm's models
still get randomness independent of every other role.
"""

from __future__ import annotations

import hashlib

import torch

# torch.manual_seed accepts the full 64-bit range, but keeping sub-seeds
# non-negative avoids surprises anywhere they are printed or re-used.
_SEED_BITS = 63


def derive_seed(seed: int, stream: str) -> int:
    """Sub-seed for one named stream: deterministic in ``seed``, independent per name."""
    digest = hashlib.blake2b(f"{seed}:{stream}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> (64 - _SEED_BITS)


def seed_stream(seed: int, stream: str) -> None:
    """Reset the global torch RNG to the start of a named stream.

    Used at phase boundaries (model construction, each training loop, the
    evaluation block) so a phase's randomness does not depend on how much the
    phases before it happened to consume.
    """
    torch.manual_seed(derive_seed(seed, stream))


def stream_generator(
    seed: int, stream: str, device: str | torch.device = "cpu"
) -> torch.Generator:
    """A standalone generator for a named stream.

    Unlike :func:`seed_stream` this is unaffected by global reseeding, which is
    what the data sampler needs: its draws must not shift when a model happens
    to be built or a phase happens to be reseeded between two batches.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(derive_seed(seed, stream))
    return generator