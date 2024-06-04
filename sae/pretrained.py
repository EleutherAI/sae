from dataclasses import dataclass
from functools import cache
from importlib import resources
from tqdm.auto import tqdm

from typing import Optional
import os

from huggingface_hub import hf_hub_download

import yaml


@dataclass
class PretrainedSAELookup:
    release: str
    repo_id: str
    model: str
    conversion_func: str | None
    saes_map: dict[str, str]  # id -> path


@cache
def get_pretrained_saes_directory() -> dict[str, PretrainedSAELookup]:
    package = "sae"
    # Access the file within the package using importlib.resources
    directory: dict[str, PretrainedSAELookup] = {}
    with resources.open_text(package, "pretrained_saes.yaml") as file:
        # Load the YAML file content
        data = yaml.safe_load(file)
        for release, value in data["SAE_LOOKUP"].items():
            saes_map: dict[str, str] = {}
            for hook_info in value["saes"]:
                saes_map[hook_info["id"]] = hook_info["path"]
            directory[release] = PretrainedSAELookup(
                release=release,
                repo_id=value["repo_id"],
                model=value["model"],
                conversion_func=value.get("conversion_func"),
                saes_map=saes_map,
            )
    return directory


def download_sae_from_hf(
    repo_id: str = "jbloom/GPT2-Small-SAEs-Reformatted",
    folder_name: str = "blocks.0.hook_resid_pre",
    force_download: bool = False,
) -> tuple[str, str, Optional[str]]:

    FILENAME = f"{folder_name}/cfg.json"
    cfg_path = hf_hub_download(
        repo_id=repo_id, filename=FILENAME, force_download=force_download
    )

    FILENAME = f"{folder_name}/sae_weights.safetensors"
    sae_path = hf_hub_download(
        repo_id=repo_id, filename=FILENAME, force_download=force_download
    )

    try:
        FILENAME = f"{folder_name}/sparsity.safetensors"
        sparsity_path = hf_hub_download(
            repo_id=repo_id, filename=FILENAME, force_download=force_download
        )
    except:  # noqa
        sparsity_path = None

    return cfg_path, sae_path, sparsity_path


def get_gpt2_res_jb_saes(
    hook_point: Optional[str] = None,
    device: str = "cpu",
) -> dict:
    """
    Download the sparse autoencoders for the GPT2-Small model with residual connections
    from the repository of jbloom. You can specify a hook_point to download only one
    of the sparse autoencoders if desired.

    """
    from .sae import SparseAutoencoder

    GPT2_SMALL_RESIDUAL_SAES_REPO_ID = "jbloom/GPT2-Small-SAEs-Reformatted"
    GPT2_SMALL_RESIDUAL_SAES_HOOK_POINTS = [
        f"blocks.{layer}.hook_resid_pre" for layer in range(12)
    ] + ["blocks.11.hook_resid_post"]

    if hook_point is not None:
        assert hook_point in GPT2_SMALL_RESIDUAL_SAES_HOOK_POINTS, (
            f"hook_point must be one of {GPT2_SMALL_RESIDUAL_SAES_HOOK_POINTS}"
            f"but got {hook_point}"
        )
        GPT2_SMALL_RESIDUAL_SAES_HOOK_POINTS = [hook_point]

    saes = {}
    for hook_point in tqdm(GPT2_SMALL_RESIDUAL_SAES_HOOK_POINTS):

        _, sae_path, _ = download_sae_from_hf(
            repo_id=GPT2_SMALL_RESIDUAL_SAES_REPO_ID, folder_name=hook_point
        )

        # Then use our function to download the files
        folder_path = os.path.dirname(sae_path)
        sae = SparseAutoencoder.load_from_pretrained(folder_path, device=device)

        saes[hook_point] = sae

    return saes
