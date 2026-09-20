"""Roofline sizing of sparse-MoE training steps; source of the paper's compute table."""

from dataclasses import dataclass

WEIGHT_BYTES = 2  # bf16 weights and gradients
WEIGHT_PASSES = 3  # forward read, activation-gradient read, weight-gradient write
OPTIMIZER_BYTES = 30  # per parameter and optimizer step: fp32 master, two moments, gradient
TARGET_TOKENS = 30e9
TARGET_TOTAL = 200e6
TARGET_ACTIVE = 25e6
MFU = 0.25
OPTIMIZER_BATCH = 2**17


@dataclass(frozen=True)
class Gpu:
    name: str
    tflops: float
    gb_per_s: float

    @property
    def ridge(self) -> float:
        """Ridge point in FLOP/byte."""
        return self.tflops * 1e12 / (self.gb_per_s * 1e9)


GPUS = (Gpu("T4", 65, 320), Gpu("A100 40GB", 312, 1555), Gpu("H100 SXM5", 989, 3350))


def active_params(n_total: float, n_expert: float, experts: int, top_k: int) -> float:
    """Dense parameters plus the routed fraction k/E of the expert parameters."""
    return n_total - n_expert + n_expert * top_k / experts


def train_flops(n_active: float, tokens: float) -> float:
    """Training FLOPs, 6 N D (forward 2, backward 4 per active parameter and token)."""
    return 6 * n_active * tokens


def intensity(n_active: float, n_total: float, tokens: float) -> float:
    """FLOP/byte of one microbatch touching every expert, counting weight traffic only."""
    return 6 * n_active * tokens / (WEIGHT_PASSES * WEIGHT_BYTES * n_total)


def min_microbatch(n_active: float, n_total: float, ridge: float) -> float:
    """Smallest microbatch, in tokens, whose intensity reaches the ridge point."""
    return ridge * WEIGHT_PASSES * WEIGHT_BYTES * n_total / (6 * n_active)


def tokens_per_second(gpu: Gpu, n_active: float, mfu: float) -> float:
    """Compute-bound throughput at a given model FLOP utilization."""
    return mfu * gpu.tflops * 1e12 / (6 * n_active)


def days(tokens: float, tok_per_s: float) -> float:
    """Wall-clock days to process `tokens` at `tok_per_s`."""
    return tokens / tok_per_s / 86400


def optimizer_overhead(gpu: Gpu, n_active: float, n_total: float, tokens: float) -> float:
    """Memory-bound optimizer time over compute time for one optimizer step."""
    t_opt = OPTIMIZER_BYTES * n_total / (gpu.gb_per_s * 1e9)
    t_compute = train_flops(n_active, tokens) / (MFU * gpu.tflops * 1e12)
    return t_opt / t_compute


def latex_rows() -> str:
    """Body rows of the compute table in the paper, one per GPU."""
    rows = []
    for gpu in GPUS:
        tps = tokens_per_second(gpu, TARGET_ACTIVE, MFU)
        cells = (
            gpu.name,
            f"{gpu.tflops:g}",
            f"{gpu.gb_per_s:,.0f}",
            f"{gpu.ridge:.0f}",
            f"{min_microbatch(TARGET_TOTAL, TARGET_TOTAL, gpu.ridge):,.0f}",
            f"{min_microbatch(TARGET_ACTIVE, TARGET_TOTAL, gpu.ridge):,.0f}",
            f"{tps:,.0f}",
            f"{days(TARGET_TOKENS, tps):.2f}",
        )
        rows.append(" & ".join(cells) + r" \\")
    return "\n".join(rows) + "\n"


def latex_tabular() -> str:
    """Complete tabular of the compute table, as included by the paper."""
    header = (
        r"GPU & $\pi$ (TFLOPS) & $b_s$ (GB/s) & $\mathcal{I}^\ast$ & $T_{\min}^{\text{dense}}$"
        r" & $T_{\min}^{\text{MoE}}$ & tok/s & Days \\"
    )
    begin = r"\begin{tabular}{lrrrrrrr}"
    return f"{begin}\n\\toprule\n{header}\n\\midrule\n{latex_rows()}\\bottomrule\n\\end{{tabular}}\n"


if __name__ == "__main__":
    print(latex_tabular(), end="")
