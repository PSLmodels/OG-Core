"""Check that pyGAM is required only by its optional fitting method."""

import builtins
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import cloudpickle
import numpy as np
import pytest

from ogcore import txfunc


def test_core_and_eilers_work_without_importing_pygam():
    """Use a fresh interpreter to catch imports anywhere in ogcore's API."""
    code = textwrap.dedent(
        """
        import sys

        class NoPyGAM:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "pygam" or fullname.startswith("pygam."):
                    raise AssertionError(
                        "Unexpected pyGAM import: " + fullname
                    )

        sys.meta_path.insert(0, NoPyGAM())
        import ogcore
        from ogcore import txfunc
        import numpy as np

        x = np.linspace(1, 10, 16)
        weights = np.linspace(1, 2, len(x))
        y = 0.1 + 0.01 * x
        rates = txfunc.get_tax_rates(
            [0.2], x, x, weights, "linear", "etr"
        )
        np.testing.assert_allclose(rates, 0.2)
        predict, fitted, error, _, _ = txfunc.monotone_spline(x, y, weights)
        np.testing.assert_allclose(fitted, y, atol=1e-7)
        np.testing.assert_allclose(predict(x[:-1]), y[:-1], atol=1e-7)
        assert np.all(np.isfinite(predict(x)))
        assert np.all(np.isfinite(fitted))
        assert np.isfinite(error)
        assert "pygam" not in sys.modules
        """
    )
    env = dict(os.environ, MPLBACKEND="Agg")
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
    )


def test_pygam_method_explains_how_to_install_extra(monkeypatch):
    original_import = builtins.__import__

    def missing_pygam(name, *args, **kwargs):
        if name == "pygam":
            raise ModuleNotFoundError("No module named 'pygam'", name="pygam")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_pygam)
    x = np.arange(8.0)
    with pytest.raises(ImportError, match=r"ogcore\[pygam\]") as error:
        txfunc.monotone_spline(x, x, np.ones_like(x), method="pygam")
    assert "mono2D" in str(error.value)
    assert "uv sync --extra pygam" in str(error.value)


def test_pygam_transitive_import_failure_is_not_masked(monkeypatch):
    original_import = builtins.__import__

    def broken_dependency(name, *args, **kwargs):
        if name == "pygam":
            raise ModuleNotFoundError(
                "Missing dependency of pyGAM",
                name="missing_transitive_package",
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_dependency)
    x = np.arange(8.0)
    with pytest.raises(ModuleNotFoundError) as error:
        txfunc.monotone_spline(x, x, np.ones_like(x), method="pygam")
    assert error.value.name == "missing_transitive_package"


def test_optional_pygam_fit_and_serialization():
    pytest.importorskip("pygam")
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, size=(80, 2))
    y = 0.05 + 0.02 * x[:, 0] + 0.01 * x[:, 1]
    weights = np.linspace(1, 2, len(x))
    predict, fitted, error, unconstrained, unconstrained_error = (
        txfunc.monotone_spline(
            x, y, weights, method="pygam", splines=[6, 6], incl_uncstr=True
        )
    )
    assert fitted.shape == y.shape
    assert np.all(np.isfinite(fitted))
    assert np.isfinite(error) and np.isfinite(unconstrained_error)
    assert unconstrained.shape == y.shape
    # The current contract is an increasing additive fit in both incomes.
    grid = np.linspace(0, 10, 40)
    for axis in range(2):
        points = np.full((len(grid), 2), 5.0)
        points[:, axis] = grid
        assert np.all(np.diff(predict(points)) >= -1e-7)
    restored = cloudpickle.loads(cloudpickle.dumps(predict))
    np.testing.assert_allclose(restored(x), fitted)
