import os
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
    parse_autotune_line,
    relay_autotune_output,
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


# ------------------------------------------------------------- parse_autotune_line
def test_parse_autotune_line_extracts_kernel_name():
    line = "Autotuning kernel l2norm_fwd_kernel with config BT: 8, num_warps: 1"
    assert parse_autotune_line(line) == ("kernel", "l2norm_fwd_kernel")


def test_parse_autotune_line_extracts_done_line_verbatim():
    line = "finished after 6.31s,"
    assert parse_autotune_line(line) == ("done", line)


def test_parse_autotune_line_ignores_unrelated_lines():
    assert parse_autotune_line("some unrelated log output") is None


# ------------------------------------------------------------- relay_autotune_output
def test_relay_autotune_output_relays_real_lines_for_arbitrary_code(monkeypatch):
    fake = _FakeTqdmFactory()
    monkeypatch.setattr("tqdm.auto.tqdm", fake)

    with relay_autotune_output("memory measurement"):
        print("Autotuning kernel l2norm_fwd_kernel with config BT: 8")

    assert "l2norm_fwd_kernel" in fake.created[0].desc


def test_relay_autotune_output_restores_stdout_even_on_exception(monkeypatch, capsys):
    fake = _FakeTqdmFactory()
    monkeypatch.setattr("tqdm.auto.tqdm", fake)

    with pytest.raises(RuntimeError), relay_autotune_output("memory measurement"):
        raise RuntimeError("boom")

    print("back to normal")
    assert "back to normal" in capsys.readouterr().out


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


def test_benchmark_step_time_returns_none_for_zero_steps_instead_of_crashing():
    calls = []

    def step_fn():
        calls.append(1)

    assert benchmark_step_time(step_fn, n_steps=0, warmup=0) is None
    assert calls == []  # never even attempted a step


def test_benchmark_step_time_survives_a_broken_bar_without_breaking_step_fn(monkeypatch):
    class _BrokenBar(_FakeBar):
        def set_description(self, desc):
            raise RuntimeError("display is broken")

    monkeypatch.setattr("tqdm.auto.tqdm", lambda total, desc, **kw: _BrokenBar(total, desc, **kw))
    calls = []

    def step_fn():
        calls.append(1)
        print("Autotuning kernel some_kernel with config BT: 8")

    result = benchmark_step_time(step_fn, n_steps=2, warmup=0)

    assert calls == [1, 1]  # step_fn ran fully despite the display raising internally
    assert result is not None and result >= 0


def test_benchmark_step_time_shows_a_tqdm_bar_instead_of_raw_logs(monkeypatch):
    fake = _FakeTqdmFactory()
    monkeypatch.setattr("tqdm.auto.tqdm", fake)

    def step_fn():
        pass

    benchmark_step_time(step_fn, n_steps=3, warmup=2)

    assert fake.created[0].total == 5  # warmup + n_steps
    assert fake.created[0].n == 5
    assert fake.created[0].closed


def test_benchmark_step_time_ticks_elapsed_time_before_any_real_output(monkeypatch):
    import kairos.utils as utils_module

    monkeypatch.setattr(utils_module, "_LOADING_TICK_SEC", 0.02)
    fake = _FakeTqdmFactory()
    monkeypatch.setattr("tqdm.auto.tqdm", fake)

    def step_fn():
        time.sleep(0.08)  # long enough for the 0.02s ticker to fire at least once

    benchmark_step_time(step_fn, n_steps=1, warmup=0)

    assert any("loading triton" in (p or "") for p in [fake.created[0].postfix_str])


def test_benchmark_step_time_stops_ticking_once_real_output_seen(monkeypatch):
    import kairos.utils as utils_module

    monkeypatch.setattr(utils_module, "_LOADING_TICK_SEC", 0.02)
    fake = _FakeTqdmFactory()
    monkeypatch.setattr("tqdm.auto.tqdm", fake)

    def step_fn():
        print("Autotuning kernel real_kernel with config BT: 8")
        time.sleep(0.08)  # ticker keeps firing after this, but must not overwrite real info

    benchmark_step_time(step_fn, n_steps=1, warmup=0)

    assert "real_kernel" in fake.created[0].desc


def test_benchmark_step_time_relays_real_triton_kernel_lines_into_bar_description(monkeypatch):
    fake = _FakeTqdmFactory()
    monkeypatch.setattr("tqdm.auto.tqdm", fake)

    def step_fn():
        print("Autotuning kernel l2norm_fwd_kernel with config BT: 8, num_warps: 1")

    benchmark_step_time(step_fn, n_steps=1, warmup=0)

    assert "l2norm_fwd_kernel" in fake.created[0].desc


def test_benchmark_step_time_relays_real_triton_finished_line_into_postfix(monkeypatch):
    fake = _FakeTqdmFactory()
    monkeypatch.setattr("tqdm.auto.tqdm", fake)

    def step_fn():
        print("finished after 6.31s,")

    benchmark_step_time(step_fn, n_steps=1, warmup=0)

    assert "finished after 6.31s" in fake.created[0].postfix_str


def test_benchmark_step_time_does_not_leak_triton_lines_to_real_stdout(capsys):
    def step_fn():
        print("Autotuning kernel foo_kernel with config BT: 8")

    benchmark_step_time(step_fn, n_steps=1, warmup=0)

    assert "Autotuning kernel" not in capsys.readouterr().out


def test_benchmark_step_time_does_not_touch_triton_env_var(monkeypatch):
    monkeypatch.setenv("TRITON_PRINT_AUTOTUNING", "1")

    def step_fn():
        pass

    benchmark_step_time(step_fn, n_steps=1, warmup=0)

    assert os.environ["TRITON_PRINT_AUTOTUNING"] == "1"  # respects the user's own setting, untouched


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


def test_training_summary_str_omits_backend_section_when_unset():
    model = nn.Linear(4, 2)
    loader = _TinyLoader(range(4))
    summary = training_summary(model, loader, epochs=1, step_fn=None)
    assert "Compute backends" not in str(summary)


def test_training_summary_str_shows_fused_backend_without_warning():
    summary = TrainingSummary(
        total_params=10,
        trainable_params=10,
        active_params=10,
        param_memory_mb=0.0,
        optimizer_memory_mb=0.0,
        total_memory_mb=0.0,
        steps_per_epoch=1,
        epochs=1,
        total_steps=1,
        attn_impl="flex",
        delta_rule_backend="fla",
        causal_conv1d_backend="causal_conv1d",
    )
    text = str(summary)
    assert "Compute backends" in text
    assert "Attention:           flex" in text
    assert "DeltaNet:            fla" in text
    assert "pip install" not in text


def test_training_summary_str_warns_on_slow_deltanet_fallback():
    summary = TrainingSummary(
        total_params=10,
        trainable_params=10,
        active_params=10,
        param_memory_mb=0.0,
        optimizer_memory_mb=0.0,
        total_memory_mb=0.0,
        steps_per_epoch=1,
        epochs=1,
        total_steps=1,
        attn_impl="flex",
        delta_rule_backend="torch_fallback",
        causal_conv1d_backend="torch_fallback",
    )
    text = str(summary)
    assert "DeltaNet:            torch_fallback  <- pip install -e '.[fast-attn]'" in text
    assert "Causal conv1d:       torch_fallback  <- pip install -e '.[fast-attn]'" in text


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


def test_training_summary_str_notes_single_gpu_benchmark_when_flagged():
    model = nn.Linear(4, 2)
    loader = _TinyLoader(range(10))

    def step_fn():
        time.sleep(0.001)

    summary = training_summary(model, loader, epochs=2, step_fn=step_fn, n_bench_steps=3)
    summary.n_gpus = 2
    summary.single_gpu_benchmark = True
    text = str(summary)
    assert "benchmarked on 1 GPU" in text


def test_training_summary_str_omits_single_gpu_note_by_default():
    model = nn.Linear(4, 2)
    loader = _TinyLoader(range(10))

    def step_fn():
        time.sleep(0.001)

    summary = training_summary(model, loader, epochs=2, step_fn=step_fn, n_bench_steps=3)
    text = str(summary)
    assert "benchmarked on 1 GPU" not in text


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
    def __init__(self, total, desc, leave=True, bar_format=None):
        self.total = total
        self.desc = desc
        self.n = 0
        self.postfix = None
        self.postfix_str = None
        self.closed = False

    def set_postfix(self, **kw):
        self.postfix = kw

    def set_postfix_str(self, s):
        self.postfix_str = s

    def set_description(self, desc):
        self.desc = desc

    def update(self, n=1):
        self.n += n

    def refresh(self):
        pass

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def test_make_progress_callback_updates_bar(monkeypatch):
    fake = _FakeTqdmFactory()
    monkeypatch.setattr("tqdm.auto.tqdm", fake)

    callback = make_progress_callback(desc="training")
    callback(1, 10, 0.5)
    callback(5, 10, 0.3)

    bar = fake.created[0]
    assert bar.total == 10
    assert bar.n == 5
    assert bar.postfix == {"loss": "0.3000"}
    assert not bar.closed


def test_make_progress_callback_closes_bar_at_last_step(monkeypatch):
    fake = _FakeTqdmFactory()
    monkeypatch.setattr("tqdm.auto.tqdm", fake)

    callback = make_progress_callback()
    callback(1, 3, 1.0)
    callback(3, 3, 0.1)

    assert fake.created[0].closed


def test_make_progress_callback_includes_stage_in_postfix_when_stage_fn_given(monkeypatch):
    fake = _FakeTqdmFactory()
    monkeypatch.setattr("tqdm.auto.tqdm", fake)

    callback = make_progress_callback(stage_fn=lambda step: "mae" if step < 5 else "diffusion")
    callback(1, 10, 0.5)
    callback(7, 10, 0.3)

    assert fake.created[0].postfix == {"loss": "0.3000", "stage": "diffusion"}


class _FakeTqdmFactory:
    def __init__(self):
        self.created = []
        self.written = []

    def __call__(self, total, desc, **kwargs):
        bar = _FakeBar(total, desc, **kwargs)
        self.created.append(bar)
        return bar

    def write(self, msg):
        self.written.append(msg)


def test_make_progress_callback_phase_updates_the_same_bar_description(monkeypatch):
    fake = _FakeTqdmFactory()
    monkeypatch.setattr("tqdm.auto.tqdm", fake)

    callback = make_progress_callback(desc="training")
    callback.phase("compiling")

    assert len(fake.created) == 1  # one bar, updated in place - no new line printed
    assert fake.created[0].desc == "training (compiling)"
    assert fake.written == []


def test_make_progress_callback_phase_before_any_step_then_step_reuses_same_bar(monkeypatch):
    fake = _FakeTqdmFactory()
    monkeypatch.setattr("tqdm.auto.tqdm", fake)

    callback = make_progress_callback(desc="training")
    callback.phase("build")
    callback(1, 10, 0.5)

    assert len(fake.created) == 1
    assert fake.created[0].n == 1
    assert fake.created[0].total == 10


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
