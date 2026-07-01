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

"""Generate and submit SLURM jobs for a sharded LeRobot conversion.

Writes two sbatch scripts (a shard-conversion array job and a dependent
aggregate job) and, unless ``--dry-run`` is given, submits them via ``sbatch``
so the aggregate runs after all shards finish (``afterok``). The library itself
never imports SLURM; this module only shells out to ``sbatch``.
"""

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .distributed import SUPPORTED_FORMATS


def _convert_command(python, input_path, shards_dir, args) -> str:
    parts = [
        python,
        "-m",
        "openarm_dataset.convert",
        str(input_path),
        str(shards_dir),
        "--format",
        args.format,
        "--num-shards",
        str(args.num_shards),
        "--shard-index",
        "${SLURM_ARRAY_TASK_ID}",
        "--fps",
        str(args.fps),
        "--smoothing-cutoff",
        str(args.smoothing_cutoff),
        "--train-split",
        str(args.train_split),
        "--jobs",
        "${SLURM_CPUS_PER_TASK}",
    ]
    if args.success_only:
        parts.append("--success-only")
    return " ".join(shlex.quote(p) if not p.startswith("$") else p for p in parts)


def _aggregate_command(python, shards_dir, output, args) -> str:
    parts = [
        python,
        "-m",
        "openarm_dataset.aggregate",
        str(shards_dir),
        str(output),
        "--format",
        args.format,
    ]
    return " ".join(shlex.quote(p) for p in parts)


def _sbatch_header(name: str, args, logs_dir: Path, extra: list[str]) -> list[str]:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
    ]
    if args.mem:
        lines.append(f"#SBATCH --mem={args.mem}")
    if args.time:
        lines.append(f"#SBATCH --time={args.time}")
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines.extend(extra)
    return lines


def _write_scripts(args) -> tuple[Path, Path, Path, Path]:
    input_path = Path(args.input).resolve()
    output = Path(args.output).resolve()
    shards_dir = (
        Path(args.shards_dir).resolve()
        if args.shards_dir
        else output.with_name(output.name + ".shards")
    )
    logs_dir = shards_dir / "_slurm"
    logs_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    convert_lines = _sbatch_header(
        f"{args.job_name}_convert",
        args,
        logs_dir,
        [
            f"#SBATCH --array=0-{args.num_shards - 1}",
            f"#SBATCH --output={logs_dir}/convert_%A_%a.out",
        ],
    )
    convert_lines += [
        "set -euo pipefail",
        _convert_command(python, input_path, shards_dir, args),
        "",
    ]

    aggregate_lines = _sbatch_header(
        f"{args.job_name}_aggregate",
        args,
        logs_dir,
        [f"#SBATCH --output={logs_dir}/aggregate_%j.out"],
    )
    aggregate_lines += [
        "set -euo pipefail",
        _aggregate_command(python, shards_dir, output, args),
        "",
    ]

    convert_path = logs_dir / "convert.sbatch"
    aggregate_path = logs_dir / "aggregate.sbatch"
    convert_path.write_text("\n".join(convert_lines))
    aggregate_path.write_text("\n".join(aggregate_lines))
    return convert_path, aggregate_path, shards_dir, output


def _sbatch(cmd: list[str]) -> str:
    """Run an sbatch command, surfacing its stderr on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"sbatch failed (exit {result.returncode}): {message}\n"
            f"Command: {' '.join(cmd)}\n"
            "Re-run with --dry-run to inspect the scripts, or adjust the SLURM "
            "options (e.g. drop --mem on clusters that do not schedule memory)."
        )
    return result.stdout.strip()


def _submit(convert_path: Path, aggregate_path: Path) -> None:
    if shutil.which("sbatch") is None:
        raise RuntimeError(
            "sbatch not found on PATH; use --dry-run to only write the scripts."
        )
    array_id = _sbatch(["sbatch", "--parsable", str(convert_path)])
    aggregate_id = _sbatch(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterok:{array_id}",
            str(aggregate_path),
        ]
    )
    print(f"Submitted shard-convert array job: {array_id}")
    print(f"Submitted aggregate job (afterok): {aggregate_id}")


def main():
    """CLI entry point for submitting a sharded conversion to SLURM."""
    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM jobs for a sharded LeRobot conversion"
    )
    parser.add_argument("input", type=Path, help="Path of the OpenArm dataset")
    parser.add_argument("output", type=Path, help="Path of the aggregated output")
    parser.add_argument("--format", required=True, choices=list(SUPPORTED_FORMATS))
    parser.add_argument(
        "--num-shards", required=True, type=int, help="Number of shards / array tasks"
    )
    parser.add_argument("--partition", required=True, help="SLURM partition")
    parser.add_argument(
        "--cpus-per-task", type=int, default=8, help="CPUs per shard task (default: 8)"
    )
    parser.add_argument("--mem", default=None, help="SLURM --mem (e.g. 32G)")
    parser.add_argument("--time", default=None, help="SLURM --time (e.g. 04:00:00)")
    parser.add_argument("--account", default=None, help="SLURM --account")
    parser.add_argument("--job-name", default="openarm", help="Base job name")
    parser.add_argument(
        "--shards-dir",
        default=None,
        type=Path,
        help="Directory for shard outputs (default: <output>.shards)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--smoothing-cutoff", type=float, default=1.0)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--success-only", action="store_true", default=False)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Write the sbatch scripts without submitting them",
    )

    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")

    convert_path, aggregate_path, shards_dir, output = _write_scripts(args)
    print(f"Shards dir: {shards_dir}")
    print(f"Output:     {output}")
    print(f"Wrote {convert_path}")
    print(f"Wrote {aggregate_path}")

    if args.dry_run:
        print("Dry run: not submitting. Submit with:")
        print(f"  arr=$(sbatch --parsable {convert_path})")
        print(f"  sbatch --dependency=afterok:$arr {aggregate_path}")
        return

    _submit(convert_path, aggregate_path)


if __name__ == "__main__":
    main()
