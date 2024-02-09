from pathlib import Path

import joblib
import torch
from s3prl.nn import S3PRLUpstream


def load_model(state, model_dir, ckpt_dir=None):
    # state_ = joblib.load(Path(model_dir,'state.pkl'))
    # if 'model' in state_:
    #     model = state_['model']
    # else:
    #     raise Exception('Model not in state')

    # if ckpt_dir is not None:
    #     model.load_state_dict(torch.load(ckpt_dir)['state_dict'],strict=False)
    # else:
    #     raise Exception('Model not found. Try rerunning the experiment to get the model in the state file.')

    upstream_model = S3PRLUpstream("wavlm_base_plus")
    state['model'] = model
    return state
