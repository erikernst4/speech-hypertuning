from itertools import zip_longest
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from loguru import logger


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


def dataset_random_split(
    original_df: pd.DataFrame, proportions: Dict[str, Union[int, float]] = {}
) -> Dict[str, pd.DataFrame]:
    remainder_k = process_proportions(proportions)

    numerical_value = any(v > 1 for v in proportions.values())
    if numerical_value:
        prop_type = 'n'
    else:
        prop_type = 'prop'

    df = original_df.copy()

    partitions_dfs: Dict[str, List[pd.DataFrame]] = {}
    for k, v in proportions.items():
        partitions_dfs[k] = []
        if k != remainder_k:
            for speaker_id in df["speaker_id"].unique():
                speaker_df = df[df.speaker_id == speaker_id].copy()

                if prop_type == 'prop':
                    sample_size = int(len(speaker_df) * v)
                else:
                    sample_size = int(v)

                sampled_idxs = np.random.choice(
                    a=speaker_df.index, size=sample_size, replace=False
                )
                speaker_partition_df = speaker_df.loc[sampled_idxs]
                df = df[~df.index.isin(speaker_partition_df.index)]  # Remove chosen

                partitions_dfs[k].append(speaker_partition_df)

    partitions = {}
    for k, v in proportions.items():
        if partitions_dfs[k]:
            partitions[k] = pd.concat(partitions_dfs[k])

    if remainder_k is not None:
        partitions[remainder_k] = df

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


def create_splits(
    state: Dict[str, Any],
    proportions: Dict[str, Any],
    output_dir: str,
    cache: bool = True,
) -> Dict[str, Any]:
    logger.info("Creating splits")

    if "splits_created" in state and state["splits_created"] and cache:
        logger.info("Splits already created, using cached splits")
        return state

    df = state['dataset_metadata'].copy()

    # Sort by audio count alternating by gender (so that taking in order is gender-balanced)
    alternated_list = get_sorted_sid_list_by_audio_count_alternating_by_gender(df=df)

    remainder_k = process_proportions(proportions)

    dfs = []
    for speaker_id in tqdm(alternated_list):
        # Process sample sizes for speaker id once, because the df will get modified
        speaker_df = df[df.speaker_id == speaker_id].copy()
        sample_sizes = process_sample_sizes(proportions=proportions, speaker_df=speaker_df)

        for partition, sample_size in sample_sizes.items():
            if partition != remainder_k:
                speaker_df = df[df.speaker_id == speaker_id].copy()
                speaker_df.drop("Set", axis=1, inplace=True)

                sampled_idxs = np.random.choice(
                    a=speaker_df.index, size=sample_size, replace=False
                )

                sampled_idxs_order = sort_idx_alternating_video_ids(speaker_df=speaker_df, sampled_idxs=sampled_idxs, sample_size=sample_size)

                assert len(sampled_idxs) == len(sampled_idxs_order) == sample_size

                speaker_partition_df = speaker_df.loc[sampled_idxs_order]

                df = df[~df.index.isin(speaker_partition_df.index)]  # Remove chosen
                speaker_partition_df['set'] = partition
                dfs.append(speaker_partition_df)


        if remainder_k is not None:
            speaker_df = df[df.speaker_id == speaker_id].copy() # Remaining speaker df
            sampled_idxs_order = sort_idx_alternating_video_ids(speaker_df=speaker_df, sampled_idxs=speaker_df.index, sample_size=sample_sizes[remainder_k])
            speaker_partition_df = speaker_df.loc[sampled_idxs_order]
            speaker_partition_df.drop("Set", axis=1, inplace=True)
            speaker_partition_df['set'] = remainder_k
            df = df[~df.index.isin(speaker_partition_df.index)]  # Remove chosen
            dfs.append(speaker_partition_df)

    splits_df = pd.concat(dfs)
    assert len(splits_df) == len(state['dataset_metadata'])
    splits_df.to_csv(output_dir + "splits.csv")

    state["dataset_metadata"] = splits_df
    state["splits_created"] = True

    return state


def process_proportions(proportions: Dict[str, Any]) -> Tuple[str, Optional[str]]:

    # Check if remainder is used only once
    remainder_splits = [k for k, v in proportions.items() if v == -1]
    if len(remainder_splits) > 1:
        raise ValueError("-1 can't be used in more than one entry")
    elif len(remainder_splits) == 1:
        remainder_k = remainder_splits[0]
    else:
        remainder_k = None

    return remainder_k


def sort_idx_alternating_video_ids(
    speaker_df: pd.DataFrame,
    sampled_idxs: np.ndarray,
    sample_size: int,
) -> List[int]:
    """
    Sort indexes alternating one from each video id
    The reason of this is to take rows in order when subsampling,
    as having different videos is more valuable than multiple
    segments from the same video_id
    """
    sampled_idxs_order = []
    unorder_speaker_partition_df = speaker_df.loc[sampled_idxs].copy()
    while len(sampled_idxs_order) < sample_size:
        for video_id in unorder_speaker_partition_df.video_id.unique():
            df_to_choose = unorder_speaker_partition_df[(unorder_speaker_partition_df.video_id == video_id) & (~unorder_speaker_partition_df.index.isin(sampled_idxs_order))]
            if df_to_choose.empty:
                continue
            idx = np.random.choice(
                a=df_to_choose.index, size=1, replace=False
            )[0]
            sampled_idxs_order.append(idx)
    return sampled_idxs_order

def process_sample_sizes(
    proportions: Dict[str, Union[int, float]],
    speaker_df: pd.DataFrame,
) -> Dict[str, int]:
    sample_sizes = {}
    for partition, v in proportions.items():
        if isinstance(v, float):
            sample_size = int(len(speaker_df) * v)
        elif isinstance(v, int):
            sample_size = int(v)
        else:
            raise ValueError(f"Unsoported value in proportions for partition {partition}: {v}")
        sample_sizes[partition] = sample_size
    return sample_sizes

def get_sorted_sid_list_by_audio_count_alternating_by_gender(
    df: pd.DataFrame
) -> List[str]:
    sid_to_audios_count = df.groupby(['speaker_id']).size().to_dict()
    sid_to_gender = (
        df[['speaker_id', 'Gender']]
        .drop_duplicates()
        .set_index("speaker_id")
        .to_dict()["Gender"]
    )
    sorted_sid_by_audios_count = [
        sid
        for sid, _ in sorted(
            sid_to_audios_count.items(), key=lambda tuple: tuple[1], reverse=True
        )
    ]

    sorted_male_sid_by_audios_count = [
        sid for sid in sorted_sid_by_audios_count if sid_to_gender[sid] == "m"
    ]
    sorted_female_sid_by_audios_count = [
        sid for sid in sorted_sid_by_audios_count if sid_to_gender[sid] == "f"
    ]

    alternated_list = []
    for male_sid, female_sid in zip_longest(
        sorted_male_sid_by_audios_count, sorted_female_sid_by_audios_count
    ):
        if male_sid is not None:
            alternated_list.append(male_sid)
        if female_sid is not None:
            alternated_list.append(female_sid)

    return alternated_list