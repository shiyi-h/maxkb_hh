import xgboost as xgb
import os 
import torch 
from huggingface_hub import snapshot_download


def get_project_base_directory():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            '../'))
    return path


def get_model():
    updown_cnt_mdl = xgb.Booster()
    if torch.cuda.is_available():
        updown_cnt_mdl.set_param({"device": "cuda"})
    try:
        model_dir = os.path.join(
            get_project_base_directory(),
            "model")
        updown_cnt_mdl.load_model(os.path.join(
            model_dir, "updown_concat_xgb.model"))
    except Exception as e:
        model_dir = snapshot_download(
            repo_id="InfiniFlow/text_concat_xgb_v1.0",
            local_dir=os.path.join(get_project_base_directory(), "model"),
            local_dir_use_symlinks=False)
        updown_cnt_mdl.load_model(os.path.join(
            model_dir, "updown_concat_xgb.model"))
    return updown_cnt_mdl