from kairos.notebook import train_with_progress


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
