import inspect
from pathlib import Path

import joblib
import torch
import torchinfo
from loguru import logger


def fit_model(
    state,
    trainer_cls=None,
    model_cls=None,
    from_checkpoint=None,
    cpu_threads: int = 8,
    dataloaders_key: str = 'dataloaders',
    checkpoint_folder: str = 'checkpoints',
    model_key_out: str = 'model',
    cache_model: bool = True,
    model_type: str = 'torch',
):

    if not ((model_key_out in state) and (cache_model)):
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

            trainer.fit(
                model,
                state[dataloaders_key]['train'],
                state[dataloaders_key]['validation'],
                ckpt_path=from_checkpoint,
            )

            trainer.save_checkpoint(Path(base_dir, 'last.ckpt'))
            state[
                model_key_out + '_checkpoint_dir'
            ] = trainer.checkpoint_callback.dirpath
            best_model_path = model.trainer.checkpoint_callback.best_model_path
            if (best_model_path is not None) and (best_model_path != ''):
                model.load_state_dict(torch.load(best_model_path)['state_dict'])
        elif model_type == 'sklearn':
            kwargs = {}
            if 'state' in inspect.signature(model_cls.__init__).parameters.keys():
                kwargs['state'] = state
            model = model_cls(**kwargs)
            X = []
            Y = []
            for x, y in state[dataloaders_key]['train']:
                X.append(x)
                Y.append(y)
            X = torch.cat(X, axis=0).detach().cpu().numpy()
            Y = torch.cat(Y, axis=0).detach().cpu().numpy()
            model.fit(X, Y)
            state[model_key_out] = model
    else:
        if hasattr(state[model_key_out], 'get_dataloaders'):
            kwargs = {}
            if 'state' in inspect.signature(model_cls.__init__).parameters.keys():
                kwargs['state'] = state
            model = model_cls(**kwargs)
            if hasattr(state[model_key_out], 'state_dict'):
                model_sd = state[model_key_out].state_dict()
                model.load_state_dict(model_sd)

            state[dataloaders_key] = model.get_dataloaders(state)
            if hasattr(state[model_key_out], 'optimizer'):
                del model.optimizer

        logger.info('Model is already in state. Skipping task.')

    return state
