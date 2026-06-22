
from huggingface_hub import HfApi

api = HfApi()

api.create_repo(repo_id="ffurfaro/bigbench", repo_type="dataset", exist_ok=True)

api.snapshot_download(
    repo_id="tasksource/bigbench",
    repo_type="dataset",
    local_dir="bigbench_local"
)

api.upload_folder(
    folder_path="bigbench_local",
    repo_id="ffurfaro/bigbench",
    repo_type="dataset"
)
