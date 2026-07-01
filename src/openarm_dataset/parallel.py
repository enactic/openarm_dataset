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

"""Small helpers for parallelizing the LeRobot conversion.

The conversion is embarrassingly parallel across episodes: loading/downsampling
and image-stats decoding are CPU/IO bound Python work (best run in separate
processes to sidestep the GIL), while video encoding just launches ``ffmpeg``
subprocesses (best run in threads so frame lists need not be pickled).

Both maps preserve input order so the serial and parallel results are identical.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def resolve_jobs(jobs: int | None) -> int:
    """Resolve a ``jobs`` argument to a concrete worker count.

    ``None`` or ``0`` means "use every available core". Negative values are
    treated as serial (1).
    """
    if jobs is None or jobs == 0:
        return os.cpu_count() or 1
    return max(1, jobs)


def parallel_map(
    func: Callable,
    items: Iterable,
    jobs: int | None,
    initializer: Callable | None = None,
    initargs: Sequence = (),
) -> list:
    """Map ``func`` over ``items`` across processes, preserving input order.

    When the resolved job count is ``<= 1`` (or there is at most one item) the
    work runs serially in the current process — identical results, no
    multiprocessing overhead, and clean tracebacks for debugging. In that case
    ``initializer`` (if given) is still called once so ``func`` sees the same
    process-global state it would in a worker.

    Exceptions raised by ``func`` propagate to the caller (fail fast); a partial
    conversion is not useful.
    """
    items = list(items)
    resolved = resolve_jobs(jobs)
    if resolved <= 1 or len(items) <= 1:
        if initializer is not None:
            initializer(*initargs)
        return [func(item) for item in items]
    with ProcessPoolExecutor(
        max_workers=min(resolved, len(items)),
        initializer=initializer,
        initargs=tuple(initargs),
    ) as executor:
        return list(executor.map(func, items))


def thread_map(func: Callable, items: Iterable, jobs: int | None) -> list:
    """Map ``func`` over ``items`` across threads, preserving input order.

    Intended for subprocess-bound work (``ffmpeg``) where the GIL is released
    while the external process runs, so threads give real concurrency without
    pickling closures or large frame lists.
    """
    items = list(items)
    resolved = resolve_jobs(jobs)
    if resolved <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(resolved, len(items))) as executor:
        return list(executor.map(func, items))


def ffmpeg_threads_for(jobs: int | None, num_encodes: int) -> int:
    """Threads per ``ffmpeg`` process so ``jobs`` concurrent encodes fill cores.

    With many small encodes and a large pool each process gets few threads;
    with few large encodes each process gets more, keeping total ffmpeg threads
    near the core count without oversubscribing.
    """
    if num_encodes <= 0:
        return 1
    active = min(resolve_jobs(jobs), num_encodes)
    cores = os.cpu_count() or 1
    return max(1, cores // active)
