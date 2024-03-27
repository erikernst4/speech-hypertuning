from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


def get_dataloaders(
    state,
    split_function=None,
    dataset_cls=None,
    dataloader_cls=None,
    dataset_key_in='dataset_metadata',
    dataset_key_out='datasets',
    partitions_key_out='partitions',
    dataloaders_key_out='dataloaders',
):

    if split_function is not None:
        partitions = split_function(state[dataset_key_in])
    else:
        partitions = {'train': state[dataset_key_in]}

    datasets = {
        k: dataset_cls[k](v, state) for k, v in partitions.items() if k in dataset_cls
    }
    dataloaders = {
        k: dataloader_cls[k](v) for k, v in datasets.items() if k in dataloader_cls
    }

    state[partitions_key_out] = partitions
    state[dataset_key_out] = datasets
    state[dataloaders_key_out] = dataloaders

    return state


def dataset_random_split(df: pd.DataFrame, proportions={}):
    numerical_value = any(v > 1 for v in proportions.values())
    if numerical_value:
        prop_type = 'n'
    else:
        prop_type = 'prop'

    # Check if remainder is used only once
    remainder_splits = [k for k, v in proportions.items() if v == -1]
    if len(remainder_splits) > 1:
        raise ValueError("-1 can't be used in more than one entry")
    elif len(remainder_splits) == 1:
        remainder_k = remainder_splits[0]
    else:
        remainder_k = None

    partitions_dfs: Dict[str, List[pd.DataFrame]] = {}
    chosen_idxs = []
    for k, v in proportions.items():
        partitions_dfs[k] = []
        if k != remainder_k:
            for speaker_id in df["speaker_id"].unique():
                speaker_df = df[df["speaker_id"] == speaker_id]

                if prop_type == 'prop':
                    v = int(len(speaker_df) * v)

                sampled_idxs = np.random.choice(
                    a=speaker_df.index, size=v, replace=False
                )
                partitions_dfs[k].append(speaker_df.loc[sampled_idxs])
                chosen_idxs += sampled_idxs.tolist()

    partitions = {}
    for k, v in proportions.items():
        partitions[k] = pd.concat(partitions_dfs[k])

    if remainder_k is not None:
        remainder_idxs = [index for index in df.index if index not in chosen_idxs]
        partitions[remainder_k] = df.loc[remainder_idxs]

    return partitions


def remove_long_audios(df, limit=10000):
    df = df.loc[df['duration'] < limit]
    return df


def dynamic_pad_batch(x):
    def not_discarded(x):
        if x is None:
            return False
        else:
            return not any([xi is None for xi in x.values()])

    def get_len(x):
        if x.ndim == 0:
            return 1
        else:
            return x.shape[0]

    def pad_to_len(x, max_len):
        if x.ndim == 0:
            return x
        else:
            pad_spec = ((0, max_len - x.shape[0]),) + ((0, 0),) * (x.ndim - 1)
            return np.pad(x, pad_spec)

    def to_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        else:
            if x.dtype in [
                np.float64,
                np.float32,
                np.float16,
                np.complex64,
                np.complex128,
                np.int64,
                np.int32,
                np.int16,
                np.int8,
                np.uint8,
                np.bool_,
            ]:

                return torch.from_numpy(x)
            else:
                return x

    x = [xi for xi in x if not_discarded(xi)]

    batch = {k: [np.array(xi[k]) for xi in x] for k in x[0]}
    batch_lens = {k: [get_len(x) for x in batch[k]] for k in batch.keys()}
    batch_max_lens = {k: max(v) for k, v in batch_lens.items()}
    batch = {
        k: np.stack([pad_to_len(x, batch_max_lens[k]) for x in batch[k]])
        for k in batch.keys()
    }
    batch_lens = {k + '_lens': np.array(v) for k, v in batch_lens.items()}
    batch.update(batch_lens)
    batch = {k: to_torch(v) for k, v in batch.items()}

    return batch


def compensate_lengths(df, chunk_length=None):
    if chunk_length is not None:
        map_idx = []
        for i, (_, row) in enumerate(df.iterrows()):
            map_idx.extend([i] * int(max(1, row['duration'] // chunk_length)))
        return map_idx
    else:
        return list(range(len(df)))


def process_classes(
    state: Dict[str, Any],
    dataset_name: str = "",
    dataset_key_in='dataset_metadata',
) -> Dict[str, Any]:

    speaker_id_column_name = "speaker_id"

    state[dataset_key_in][speaker_id_column_name] = state[dataset_key_in][
        speaker_id_column_name
    ].apply(
        lambda speaker_id: (
            speaker_id + "_" + dataset_name
            if dataset_name not in speaker_id
            else speaker_id
        )
    )

    mapping = {
        original_id: i
        for i, original_id in enumerate(
            state[dataset_key_in][speaker_id_column_name].unique()
        )
    }

    state[dataset_key_in]['class_id'] = state[dataset_key_in][
        speaker_id_column_name
    ].apply(lambda x: mapping[x])

    state["speaker_id_mapping"] = mapping

    return state
