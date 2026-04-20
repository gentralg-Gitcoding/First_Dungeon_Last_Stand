from huggingface_hub import upload_file

def upload_gans_model():
    upload_file(
        path_or_fileobj="game/data/models/generator_epoch_49.pth",
        path_in_repo="generator_model.pth",
        repo_id="gentralg/GANs-Dungeon-Floor-Entities",
        repo_type="model"
    )

def upload_diffusion_model():
    upload_file(
        path_or_fileobj="game/data/models/diffusion_model.pth",
        path_in_repo="diffusion_model.pth",
        repo_id="gentralg/Diffusion-Dungeon-Floor-Entities",
        repo_type="model"
    )

def main():
    upload_gans_model()
    upload_diffusion_model()

if __name__ == "__main__":
    main()