from huggingface_hub import snapshot_download

repo_id = "minchul/cvlface_adaface_vit_base_kprpe_webface12m"

local_dir = "/home/user1/Download/adaface_vit_base_kprpe_webface12m"

snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)