from pathlib import Path

import joblib
import torch


def load_model(state, model_dir, ckpt_dir=None):
    state_ = joblib.load(Path(model_dir, 'state.pkl'))
    if 'model' in state_:
        model = state_['model']
    else:
        raise ValueError('Model not in state')

    if ckpt_dir is not None:
        model.load_state_dict(torch.load(ckpt_dir)['state_dict'], strict=False)
    else:
        raise ValueError(
            'Model not found. Try rerunning the experiment to get the model in the state file.'
        )

    state['model'] = model
    return state
