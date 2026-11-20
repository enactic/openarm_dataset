# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Low-pass smoothing applied by the exporters."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openarm_dataset import Dataset

FIXTURE_DIR = Path(__file__).parent / "fixture"
FPS = 250.0


def _sine_gain(dataset: Dataset, frequency: float, cutoff: float) -> float:
    t = np.arange(0, 40, 1 / FPS)
    df = pd.DataFrame({"q": np.sin(2 * np.pi * frequency * t)}, index=t)
    filtered = dataset._apply_smoothing(df, cutoff=cutoff, fps=FPS)["q"].to_numpy()
    # Drop the filtfilt edges and measure the steady-state amplitude.
    return float(np.sqrt(2) * filtered[2000:-2000].std())


@pytest.fixture
def dataset() -> Dataset:
    return Dataset(FIXTURE_DIR / "dataset_0.4.0_qpos")


@pytest.mark.parametrize("cutoff", [0.5, 1.0, 1.25, 5.0])
def test_cutoff_is_applied_as_given(dataset, cutoff):
    # A zero-phase 4th-order Butterworth has gain 0.5 at its cutoff frequency.
    assert _sine_gain(dataset, cutoff, cutoff) == pytest.approx(0.5, abs=0.01)


def test_cutoff_at_or_above_nyquist_is_rejected(dataset):
    df = pd.DataFrame({"q": np.zeros(100)})
    with pytest.raises(ValueError, match="Nyquist"):
        dataset._apply_smoothing(df, cutoff=FPS / 2, fps=FPS)
