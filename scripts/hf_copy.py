import shutil
import tempfile
from huggingface_hub import HfApi, snapshot_download, upload_folder

api = HfApi()

datasets = [
    ("HuggingFaceTB/cosmopedia","ffurfaro/cosmopedia"),
    ("tasksource/bigbench", "ffurfaro/bigbench"),
    ("yahma/alpaca-cleaned", "ffurfaro/alpaca-cleaned"),
    ("Team-ACE/ToolACE", "ffurfaro/ToolACE"),
    ("argilla/ultrafeedback-binarized-preferences-cleaned", "ffurfaro/ultrafeedback-cleaned"),
]

# delete repos
for _, dst in datasets:
    try:
        api.delete_repo(repo_id=dst, repo_type="dataset")
        print(f"Deleted: {dst}")
    except:
        pass

# clean upload
for src, dst in datasets:
    tmp = tempfile.mkdtemp()
    print(f"{src} -> {dst}")

    snapshot_download(repo_id=src, repo_type="dataset", local_dir=tmp)
    api.create_repo(repo_id=dst, repo_type="dataset", exist_ok=True)
    upload_folder(folder_path=tmp, repo_id=dst, repo_type="dataset")

    shutil.rmtree(tmp)  # critical

    print(f"tmp folder deleted: {tmp}")
    print(f"✅ Done: {dst}")