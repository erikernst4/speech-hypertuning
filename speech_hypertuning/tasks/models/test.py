import inspect
from pathlib import Path
from typing import Any, Dict

import joblib
import torch
import torchinfo
from loguru import logger


def test_model(
    state: Dict[str, Any],
    trainer_cls=None,
    model_cls=None,
    from_checkpoint='last',
    cpu_threads: int = 8,
    dataloaders_key: str = 'dataloaders',
    checkpoint_folder: str = 'checkpoints',
    model_type: str = 'torch',
):

    if model_type == 'torch':
        torch.set_num_threads(cpu_threads)
        torch.set_float32_matmul_precision('medium')
        kwargs = {}
        if 'state' in inspect.signature(model_cls.__init__).parameters.keys():
            kwargs['state'] = state

        if model_cls is None:
            if Path(from_checkpoint).stem == 'state':
                model = joblib.load(from_checkpoint)['model']
                from_checkpoint = None
        else:
            model = model_cls(**kwargs)
        trainer = trainer_cls()
        trainer.checkpoint_callback.dirpath = (
            trainer.checkpoint_callback.dirpath + '/{}'.format(checkpoint_folder)
        )
        base_dir = trainer.checkpoint_callback.dirpath
        # Find last checkpoint
        if from_checkpoint == 'last':
            ckpts = list(Path(base_dir).glob('*.ckpt'))
            if 'last' in [x.stem for x in ckpts]:
                from_checkpoint = Path(base_dir, 'last.ckpt')
            else:
                ckpt_epoch = [
                    int(c.stem.split('epoch=')[-1].split('-')[0]) for c in ckpts
                ]
                if len(ckpt_epoch) > 0:
                    last_epoch = max(ckpt_epoch)
                    from_checkpoint = ckpts[ckpt_epoch.index(last_epoch)]
                else:
                    logger.info(
                        'No checkpoints found in {}. Training from scratch.'.format(
                            base_dir
                        )
                    )
                    from_checkpoint = None

        logger.info(torchinfo.summary(model))
        if from_checkpoint is not None:
            ckpt_data = torch.load(from_checkpoint)
            model.set_optimizer_state(ckpt_data['optimizer_states'])
            model.load_state_dict(ckpt_data['state_dict'], strict=False)
            from_checkpoint = None

        test_metrics = trainer.test(
            ckpt_path=from_checkpoint,
            dataloaders=state[dataloaders_key]['validation'],
            verbose=True,
        )

        state['test_metrics'] = test_metrics

        return state
