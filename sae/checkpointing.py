import os


SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"


def save_checkpoint(
    trainer,
    checkpoint_name: int | str,
):
    for i, sae in enumerate(trainer.saes):
        checkpoint_path = f"{sae.cfg.checkpoint_path}/{checkpoint_name}_layer{i}"

        os.makedirs(checkpoint_path, exist_ok=True)

        path = f"{checkpoint_path}"
        os.makedirs(path, exist_ok=True)

        if sae.normalize_sae_decoder:
            sae.set_decoder_norm_to_unit_norm()
        sae.save_model(path)
