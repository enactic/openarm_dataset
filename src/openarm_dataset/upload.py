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

"""Upload OpenArm Dataset to Hugging Face Hub."""

import argparse
import pathlib
import shutil
import sys
import importlib.resources
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from huggingface_hub.errors import RevisionNotFoundError
import contextlib

from .dataset import Dataset


def pack_cameras_as_tar(dataset: Dataset) -> None:
    """Repack every "dir"-format camera into a sibling ".tar" archive in place.

    Each ``cameras/<name>/`` directory of JPEG frames is replaced by one
    uncompressed ``cameras/<name>.tar`` archive. Packing is lossless and
    reversible: ``Dataset`` reads either layout through the same API.

    Args:
        dataset: The dataset to repack in place.

    """
    for episode in dataset.meta.episodes:
        for camera in dataset.load_cameras(episode).values():
            if camera.format == "tar":
                continue
            camera.write(camera.base_path, "tar")
            shutil.rmtree(camera.base_path)


def create_dataset_card(
    tags: list | None = None,
    metadata_yaml: str | None = None,
    camera_names: list[str] | None = None,
    **kwargs,
) -> DatasetCard:
    """Create a `DatasetCard` for a OpenArm Dataset.

    Args:
        tags (list | None): A list of tags to add to the dataset card.
        metadata_yaml (str | None): The dataset's ``metadata.yaml`` contents,
            embedded verbatim on the card.
        camera_names (list[str] | None): Camera names to expose as dataset
            viewer configs. Each becomes a WebDataset config so the camera
            frames are browsable on the Hugging Face Hub.
        **kwargs: Additional keyword arguments to populate the card template.

    Returns:
        DatasetCard: The generated dataset card object.

    """
    card_tags = ["OpenArm"]

    if tags:
        card_tags += tags
    if kwargs.get("license"):
        kwargs = {**kwargs, "license": kwargs["license"]}
    if metadata_yaml:
        dataset_structure = "[metadata.yaml](metadata.yaml):\n"
        dataset_structure += f"```yaml\n{metadata_yaml}\n```\n"
        kwargs = {**kwargs, "dataset_structure": dataset_structure}
    configs = [
        {
            "config_name": name,
            "data_files": f"episodes/*/cameras/{name}.tar",  # for dataset viewer
        }
        for name in (camera_names or [])
    ]
    card_data = DatasetCardData(
        license=kwargs.get("license"),
        tags=card_tags,
        task_categories=["robotics"],
        configs=configs or None,
    )

    card_template = (
        importlib.resources.files("openarm_dataset")
        .joinpath("card_template.md")
        .read_text()
    )

    return DatasetCard.from_template(
        card_data=card_data, template_str=card_template, **kwargs
    )


def upload_dataset(
    input_path: pathlib.Path,
    repo_id: str,
    branch: str = "main",
    tag: str | None = None,
    metadata_yaml: str | None = None,
    licence: str | None = None,
    camera_names: list[str] | None = None,
    private: bool = False,
    upload_large_folder: bool = False,
) -> None:
    """Upload an OpenArm Dataset directory to the Hugging Face Hub.

    Creates the dataset repository if it does not exist, then uploads the whole
    directory. Camera frames are never uploaded as loose image files; pack them
    into ``.tar`` archives first (see ``Dataset.write(camera_format="tar")``).

    Args:
        input_path: Path of the OpenArm Dataset directory to upload.
        repo_id: Target repository id, e.g. ``username/dataset-name``.
        branch: Branch (revision) to upload to.
        tag: If given, create this tag on ``branch`` after the upload.
        metadata_yaml: The dataset's ``metadata.yaml`` contents, shown verbatim
            on the dataset card.
        licence: Licence identifier recorded on the dataset card.
        camera_names: Camera names to expose as dataset viewer configs so the
            camera frames are browsable on the Hugging Face Hub.
        private: Create the repository as private when it does not exist.
        upload_large_folder: Use ``upload_large_folder`` for a resumable,
            multi-threaded upload of large datasets.

    """
    hf_api = HfApi()
    hf_api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    # Never upload camera frames as loose image files; they belong in .tar
    # archives to stay within Hugging Face Hub's per-repository file-count limit.
    ignore_patterns = ["*.jpeg", "*.jpg", "*.png"]
    upload_kwargs = {
        "repo_id": repo_id,
        "folder_path": str(input_path),
        "repo_type": "dataset",
        "revision": branch,
        "ignore_patterns": ignore_patterns,
    }
    if upload_large_folder:
        hf_api.upload_large_folder(**upload_kwargs)
    else:
        hf_api.upload_folder(**upload_kwargs)

    card = create_dataset_card(
        tag=tag,
        metadata_yaml=metadata_yaml,
        license=licence,
        camera_names=camera_names,
    )
    card.push_to_hub(
        repo_id=repo_id,
        repo_type="dataset",
        revision=branch,
    )
    if tag is not None:
        with contextlib.suppress(RevisionNotFoundError):
            hf_api.delete_tag(repo_id, tag=tag, repo_type="dataset")
        hf_api.create_tag(
            repo_id, tag=tag, revision=branch, repo_type="dataset", exist_ok=True
        )


def main():
    """Upload OpenArm Dataset to Hugging Face Hub."""
    parser = argparse.ArgumentParser(
        description="Upload an OpenArm Dataset to the Hugging Face Hub"
    )
    parser.add_argument(
        "input",
        help="Path of an OpenArm Dataset to upload",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face dataset repository id, e.g. username/dataset-name",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create the repository as private if it does not exist",
    )
    parser.add_argument(
        "--licence",
        default="apache-2.0",
        help="The licence to associate with the dataset on the Hugging Face Hub. "
        "Defaults to Apache-2.0.",
    )
    parser.add_argument(
        "--large-folder",
        action="store_true",
        default=False,
        help="Use a resumable, multi-threaded upload for large datasets. "
        "Recommended for datasets larger than 1 GB.",
    )
    args = parser.parse_args()

    dataset = Dataset(args.input)

    if dataset.camera_format == "dir":
        print(
            "Packing camera frames into .tar archives in place before upload "
            "(Hugging Face Hub file-count recommendation)...",
            file=sys.stderr,
        )
        pack_cameras_as_tar(dataset)

    upload_dataset(
        args.input,
        args.repo_id,
        tag=dataset.meta.version,
        metadata_yaml=(args.input / "metadata.yaml").read_text(),
        licence=args.licence,
        camera_names=dataset.camera_names,
        upload_large_folder=args.large_folder,
        private=args.private,
    )


if __name__ == "__main__":
    main()
