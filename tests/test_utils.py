import time

import pytest
import torch
from torch import nn

from kairos.utils import (
    DetailedMemoryReport,
    TrainingSummary,
    benchmark_step_time,
    count_active_parameters,
    count_parameters,
    detailed_memory_report,
    estimate_optimizer_memory_mb,
    estimate_param_memory_mb,
    format_duration,
    locate_first_nonfinite_module,
    make_progress_callback,
    training_summary,
)


# --------------------------------------------------- locate_first_nonfinite_module
def test_locate_first_nonfinite_module_finds_the_offending_layer():
    class Bad(nn.Module):
        def forward(self, x):
            return x * float("nan")

    model = nn.Sequential(nn.Linear(4, 4), Bad(), nn.Linear(4, 4))
    x = torch.randn(2, 4)

    result = locate_first_nonfinite_module(model, lambda: model(x))

    assert result is not None
    assert result["module_type"] == "Bad"
    assert result["nan_frac"] == 1.0


def test_locate_first_nonfinite_module_returns_none_when_all_finite():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    x = torch.randn(2, 4)

    assert locate_first_nonfinite_module(model, lambda: model(x)) is None


# ------------------------------------------------------------- format_duration
@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "0s"),
        (5, "5s"),
        (65, "1m 5s"),
        (3725, "1h 2m 5s"),
        (None, "n/a"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


# ------------------------------------------------------------- count_parameters
def test_count_parameters_all_trainable():
    model = nn.Linear(4, 2)  # 4*2 + 2 = 10
    total, trainable = count_parameters(model)
    assert total == 10
    assert trainable == 10


def test_count_parameters_with_frozen_layer():
    model = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 1))
    for p in model[1].parameters():
        p.requires_grad = False
    total, trainable = count_parameters(model)
    frozen = sum(p.numel() for p in model[1].parameters())
    assert total == 10 + (2 * 1 + 1)
    assert trainable == total - frozen


# ------------------------------------------------------------- memory estimates
def test_estimate_param_memory_mb():
    # 1M fp32 params -> ~3.81 MB
    assert estimate_param_memory_mb(1_000_000) == pytest.approx(3.8147, rel=1e-3)


def test_estimate_optimizer_memory_mb_is_double_param_memory_for_adamw():
    trainable = 1_000_000
    assert estimate_optimizer_memory_mb(trainable) == pytest.approx(2 * estimate_param_memory_mb(trainable))


# ------------------------------------------------------------- benchmark_step_time
def test_benchmark_step_time_returns_positive_average():
    def step_fn():
        time.sleep(0.001)

    avg = benchmark_step_time(step_fn, n_steps=3, warmup=1)
    assert avg is not None
    assert avg > 0


def test_benchmark_step_time_returns_none_when_iterator_exhausted():
    values = iter([1, 2])  # only 2 values: warmup=1 consumes

    def step_fn():
        return next(values)

    assert benchmark_step_time(step_fn, n_steps=5, warmup=1) is None


# ------------------------------------------------------------- training_summary
class _TinyLoader(list):
    """A list is already sized and iterable, which is all training_summary needs."""


def test_training_summary_without_benchmark():
    model = nn.Linear(4, 2)
    loader = _TinyLoader(range(5))  # steps_per_epoch = 5
    summary = training_summary(model, loader, epochs=3, step_fn=None)

    assert isinstance(summary, TrainingSummary)
    assert summary.total_params == 10
    assert summary.trainable_params == 10
    assert summary.steps_per_epoch == 5
    assert summary.epochs == 3
    assert summary.total_steps == 15
    assert summary.avg_step_time_sec is None
    assert summary.estimated_total_time_sec is None


def test_training_summary_with_benchmark():
    model = nn.Linear(4, 2)
    loader = _TinyLoader(range(10))

    def step_fn():
        time.sleep(0.001)

    summary = training_summary(model, loader, epochs=2, step_fn=step_fn, n_bench_steps=3)

    assert summary.avg_step_time_sec is not None
    assert summary.avg_step_time_sec > 0
    assert summary.estimated_total_time_sec == pytest.approx(summary.avg_step_time_sec * summary.total_steps)


def test_training_summary_str_contains_key_fields():
    model = nn.Linear(4, 2)
    loader = _TinyLoader(range(4))
    summary = training_summary(model, loader, epochs=1, step_fn=None)
    text = str(summary)
    assert "Total params" in text
    assert "Active params" in text
    assert "Total steps:         4" in text
    assert "n/a" in text  # no benchmark run


def test_training_summary_with_benchmark_shows_measured_step_time_in_str():
    model = nn.Linear(4, 2)
    loader = _TinyLoader(range(10))

    def step_fn():
        time.sleep(0.001)

    summary = training_summary(model, loader, epochs=2, step_fn=step_fn, n_bench_steps=3)
    text = str(summary)
    assert "Avg step time:" in text
    assert "ms" in text
    assert "Est. total time:" in text
    assert "n/a" not in text


def test_training_summary_str_uses_measured_label_when_flag_set():
    model = nn.Linear(4, 2)
    loader = _TinyLoader(range(4))
    summary = training_summary(model, loader, epochs=1, step_fn=None)
    summary.measured_memory = True
    text = str(summary)
    assert "Measured model memory:" in text
    assert "Measured optimizer mem:" in text
    assert "Measured total memory:" in text
    assert "Est. model memory:" not in text


def test_training_summary_str_uses_est_label_by_default():
    model = nn.Linear(4, 2)
    loader = _TinyLoader(range(4))
    summary = training_summary(model, loader, epochs=1, step_fn=None)
    assert summary.measured_memory is False
    text = str(summary)
    assert "Est. model memory:" in text
    assert "Est. optimizer mem:" in text
    assert "Est. total memory:" in text


# ------------------------------------------------------------- count_active_parameters
class _MoEModule(nn.Module):
    def __init__(self, n_experts=4, dim=4):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.shared = nn.Linear(dim, dim)  # always active, not counted as
        self.mlp.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_experts)])


def test_count_active_parameters_without_moe_returns_total():
    model = nn.Linear(4, 2)
    total, _ = count_parameters(model)
    assert count_active_parameters(model) == total


def test_count_active_parameters_with_moe_is_less_than_total():
    model = _MoEModule(n_experts=4)
    total, _ = count_parameters(model)
    active = count_active_parameters(model, num_experts_per_tok=1, num_local_experts=4)
    assert active < total


def test_count_active_parameters_moe_formula():
    model = _MoEModule(n_experts=4)
    total, _ = count_parameters(model)
    expert_params = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)
    shared_params = total - expert_params
    active = count_active_parameters(model, num_experts_per_tok=2, num_local_experts=4)
    assert active == shared_params + expert_params // 2


def test_training_summary_includes_active_params_for_moe():
    model = _MoEModule(n_experts=4)
    loader = _TinyLoader(range(3))
    summary = training_summary(model, loader, epochs=1, num_experts_per_tok=1, num_local_experts=4)
    assert summary.active_params < summary.total_params


# ------------------------------------------------------------- make_progress_callback
class _FakeBar:
    def __init__(self, total, desc):
        self.total = total
        self.desc = desc
        self.n = 0
        self.postfix = None
        self.closed = False

    def set_postfix(self, **kw):
        self.postfix = kw

    def refresh(self):
        pass

    def close(self):
        self.closed = True


def test_make_progress_callback_updates_bar(monkeypatch):
    created = []
    monkeypatch.setattr("tqdm.auto.tqdm", lambda total, desc: created.append(_FakeBar(total, desc)) or created[-1])

    callback = make_progress_callback(desc="training")
    callback(1, 10, 0.5)
    callback(5, 10, 0.3)

    bar = created[0]
    assert bar.total == 10
    assert bar.n == 5
    assert bar.postfix == {"loss": "0.3000"}
    assert not bar.closed


def test_make_progress_callback_closes_bar_at_last_step(monkeypatch):
    created = []
    monkeypatch.setattr("tqdm.auto.tqdm", lambda total, desc: created.append(_FakeBar(total, desc)) or created[-1])

    callback = make_progress_callback()
    callback(1, 3, 1.0)
    callback(3, 3, 0.1)

    assert created[0].closed


# ------------------------------------------------------------- detailed_memory_report
def _tiny_model_and_optimizer():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_detailed_memory_report_without_scaler_runs_one_real_step():
    model, optimizer = _tiny_model_and_optimizer()
    x = torch.randn(2, 8)
    target = torch.randn(2, 4)

    def loss_fn():
        return torch.nn.functional.mse_loss(model(x), target)

    before = {k: v.clone() for k, v in model.state_dict().items()}
    report = detailed_memory_report(model, optimizer, loss_fn, device=torch.device("cpu"))

    assert isinstance(report, DetailedMemoryReport)
    assert report.unique_param_bytes > 0
    assert report.grad_bytes > 0
    assert report.optimizer_state_bytes > 0  # AdamW m/v buffers, present after optimizer.step()
    assert report.device == "cpu"
    # a real step ran (no scaler path): weights actually moved
    after = model.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_detailed_memory_report_with_scaler_runs_one_real_step():
    model, optimizer = _tiny_model_and_optimizer()
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)  # CPU: scaler must be disabled
    x = torch.randn(2, 8)
    target = torch.randn(2, 4)

    def loss_fn():
        return torch.nn.functional.mse_loss(model(x), target)

    report = detailed_memory_report(model, optimizer, loss_fn, device=torch.device("cpu"), scaler=scaler)

    assert report.optimizer_state_bytes > 0
    assert report.grad_bytes > 0


def test_detailed_memory_report_module_breakdown_covers_leaf_modules():
    model, optimizer = _tiny_model_and_optimizer()
    x = torch.randn(2, 8)
    target = torch.randn(2, 4)

    def loss_fn():
        return torch.nn.functional.mse_loss(model(x), target)

    report = detailed_memory_report(model, optimizer, loss_fn, device=torch.device("cpu"))
    names = {row["name"] for row in report.module_breakdown}
    assert any("0" in n for n in names)  # first Linear
    assert any("2" in n for n in names)  # second Linear


def test_detailed_memory_report_str_contains_key_fields():
    model, optimizer = _tiny_model_and_optimizer()
    x = torch.randn(2, 8)
    target = torch.randn(2, 4)

    def loss_fn():
        return torch.nn.functional.mse_loss(model(x), target)

    report = detailed_memory_report(model, optimizer, loss_fn, device=torch.device("cpu"))
    text = str(report)
    assert "RSS" in text
    assert "Unaccounted" in text
