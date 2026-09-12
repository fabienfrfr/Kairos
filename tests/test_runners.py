import contextlib

import torch

import kairos.runners as runners_mod
from kairos.runners import evaluate_and_log, overfit_with_progress, run_generation_demo, train_with_progress


class _FakeTrainConfig:
    def __init__(self, epochs):
        self.epochs = epochs


class _FakePipe:
    def __init__(self, tmp_path, resume_exists=False):
        self.ckpt_dir = tmp_path
        if resume_exists:
            (tmp_path / "last.pt").write_text("x")
        self.train_config = _FakeTrainConfig(epochs=2)
        self.loader = [0, 1, 2]
        self.train_calls = []
        self.best_loss = 0.5
        self.skipped_nonfinite_steps = 0
        self.eval_log_rows = []
        self.best_eval_loss = None
        self.curriculum_bounds = None

    def train(self, **kwargs):
        self.train_calls.append(kwargs)
        return [{"step": 1}]


class _FakeMarimoBar:
    def __init__(self):
        self.updates = []

    def update(self, increment=0, subtitle=None):
        self.updates.append((increment, subtitle))

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeMarimoStatus:
    def __init__(self, outer):
        self._outer = outer

    def progress_bar(self, total, title):
        self._outer.bar_created = True
        return self._outer.bar


class _FakeMarimo:
    def __init__(self, running_in_notebook=True):
        self._running = running_in_notebook
        self.bar = _FakeMarimoBar()
        self.status = _FakeMarimoStatus(self)
        self.bar_created = False

    def running_in_notebook(self):
        return self._running


def test_train_with_progress_prints_resuming_when_checkpoint_exists(tmp_path, capsys):
    pipe = _FakePipe(tmp_path, resume_exists=True)

    train_with_progress(pipe, force_restart=False, mo=None)

    assert "resuming" in capsys.readouterr().out


def test_train_with_progress_prints_force_restart_message(tmp_path, capsys):
    pipe = _FakePipe(tmp_path, resume_exists=True)

    train_with_progress(pipe, force_restart=True, mo=None)

    assert "FORCE_RESTART is True" in capsys.readouterr().out


def test_train_with_progress_passes_resume_flag_through_to_pipe(tmp_path):
    pipe = _FakePipe(tmp_path)

    train_with_progress(pipe, force_restart=False, mo=None)

    assert pipe.train_calls[0]["resume"] is True


def test_train_with_progress_uses_marimo_bar_when_available(tmp_path):
    pipe = _FakePipe(tmp_path)
    mo = _FakeMarimo(running_in_notebook=True)

    train_with_progress(pipe, force_restart=False, mo=mo)

    assert mo.bar_created
    assert pipe.train_calls[0]["resume"] is True


def test_train_with_progress_ignores_mo_when_not_running_in_notebook(tmp_path):
    pipe = _FakePipe(tmp_path)
    mo = _FakeMarimo(running_in_notebook=False)

    train_with_progress(pipe, force_restart=False, mo=mo)

    assert not mo.bar_created


def test_train_with_progress_prints_final_summary(tmp_path, capsys):
    pipe = _FakePipe(tmp_path)

    train_with_progress(pipe, force_restart=False, mo=None)

    out = capsys.readouterr().out
    assert "training complete - steps: 1" in out
    assert "best avg-epoch loss: 0.5000" in out
    assert f"checkpoints: {pipe.ckpt_dir}" in out


def test_train_with_progress_prints_eval_line_only_when_eval_rows_exist(tmp_path, capsys):
    pipe = _FakePipe(tmp_path)
    pipe.eval_log_rows = [{"loss": 1.0}]
    pipe.best_eval_loss = 0.9

    train_with_progress(pipe, force_restart=False, mo=None)

    assert "eval points: 1" in capsys.readouterr().out


def test_train_with_progress_returns_the_training_logs(tmp_path):
    pipe = _FakePipe(tmp_path)

    logs = train_with_progress(pipe, force_restart=False, mo=None)

    assert logs == [{"step": 1}]


class _FakeOverfitPipe:
    def __init__(self):
        self.overfit_calls = []
        self.curriculum_bounds = None

    def overfit_test(self, **kwargs):
        self.overfit_calls.append(kwargs)
        if kwargs.get("progress_callback") is not None:
            kwargs["progress_callback"](1, kwargs["steps"], 1.0)
            kwargs["progress_callback"](2, kwargs["steps"], 0.1)
        return [{"loss": 1.0}, {"loss": 0.1}]


def test_overfit_with_progress_prints_loss_summary(capsys):
    pipe = _FakeOverfitPipe()

    overfit_with_progress(pipe, n_examples=16, steps=200, log_every=10, mo=None)

    assert "1.0000 -> 0.1000" in capsys.readouterr().out


def test_overfit_with_progress_uses_marimo_bar_when_available():
    pipe = _FakeOverfitPipe()
    mo = _FakeMarimo(running_in_notebook=True)

    overfit_with_progress(pipe, n_examples=16, steps=200, log_every=10, mo=mo)

    assert mo.bar_created


def test_overfit_with_progress_returns_the_logs():
    pipe = _FakeOverfitPipe()

    logs = overfit_with_progress(pipe, n_examples=16, steps=200, log_every=10, mo=None)

    assert logs == [{"loss": 1.0}, {"loss": 0.1}]


def test_overfit_with_progress_announces_curriculum_stage_changes(capsys):
    pipe = _FakeOverfitPipe()
    pipe.curriculum_bounds = (1, 0)  # step 0 is 'mae', step >= 1 is 'diffusion' (no transition)

    overfit_with_progress(pipe, n_examples=16, steps=200, log_every=10, mo=None)

    out = capsys.readouterr().out
    assert "entering 'mae' stage" not in out  # first callback call is at step=1, already past mae_steps
    assert "entering 'diffusion' stage" in out


def test_overfit_with_progress_reports_fixed_regime_when_no_curriculum_bounds(capsys):
    pipe = _FakeOverfitPipe()
    pipe.curriculum_bounds = None

    overfit_with_progress(pipe, n_examples=16, steps=200, log_every=10, mo=None)

    assert "entering 'fixed regime' stage" in capsys.readouterr().out


class _FakeEvalDataConfig:
    def __init__(self, multimodal_examples):
        self.multimodal_examples = multimodal_examples
        self.max_len = 8
        self.stride = 1
        self.batch_size = 2


class _FakeWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


class _FakeEvalPipe:
    def __init__(self):
        self.device = "cpu"
        self.global_step = 3
        self.writer = _FakeWriter()
        self.eval_calls = 0
        self.tokenizer = None

    def _autocast(self):
        return contextlib.nullcontext()

    class hf_trainer:
        @staticmethod
        def compute_loss(model, batch):
            return batch["loss"].clone()

    def model_eval(self):
        pass


def test_evaluate_and_log_skips_when_no_eval_examples(capsys):
    pipe = _FakeEvalPipe()

    result = evaluate_and_log(pipe, _FakeEvalDataConfig(multimodal_examples=[]))

    assert result is None
    assert "no eval examples" in capsys.readouterr().out


def test_evaluate_and_log_averages_loss_and_logs_to_writer(monkeypatch, capsys):
    pipe = _FakeEvalPipe()
    pipe.model = type("M", (), {"eval": lambda self: None, "train": lambda self: None})()
    fake_batches = [{"loss": torch.tensor(2.0)}, {"loss": torch.tensor(4.0)}]
    monkeypatch.setattr(runners_mod, "KairosPretrainingDataset", lambda **kwargs: [1, 2, 3, 4])
    monkeypatch.setattr(runners_mod, "DataLoader", lambda dataset, batch_size, shuffle: fake_batches)

    eval_loss = evaluate_and_log(pipe, _FakeEvalDataConfig(multimodal_examples=["a", "b"]))

    assert eval_loss == 3.0
    assert pipe.writer.scalars == [("eval/loss", 3.0, 3)]
    assert "eval loss: 3.0000" in capsys.readouterr().out


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, ids, skip_special_tokens=True):
        return f"decoded:{ids}"


class _FakeGenPipe:
    def generate(self, prompt, **kwargs):
        return list(prompt) + [99, 99]


def test_run_generation_demo_falls_back_to_text_examples_when_eval_too_short():
    tokenizer = _FakeTokenizer()
    pipe = _FakeGenPipe()
    eval_examples = [{"modality": "text", "text": "too short"}]
    text_examples = [{"modality": "text", "text": "a much longer prompt sentence here"}]

    results = run_generation_demo(pipe, tokenizer, eval_examples, text_examples, 2, 2, 4, 0.5, 0.4, 1.0, 0, 1)

    assert len(results) == 1
    assert results[0]["prompt"].startswith("decoded:")


def test_run_generation_demo_returns_empty_when_no_rows_qualify():
    tokenizer = _FakeTokenizer()
    pipe = _FakeGenPipe()

    results = run_generation_demo(pipe, tokenizer, [], [], 5, 2, 4, 0.5, 0.4, 1.0, 0, 1)

    assert results == []
