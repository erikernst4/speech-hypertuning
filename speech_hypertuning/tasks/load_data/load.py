import re
from pathlib import Path
from random import sample
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
from loguru import logger
from tqdm import tqdm

from speech_hypertuning.tasks.load_data.process import process_sample_sizes


def load_dataset(
    state: Dict[str, Any],
    reader_fn,
    cache: bool = True,
    filters: List[Any] = [],
    key_out: str = 'dataset_metadata',
    splits_csv: Optional[str] = None,
    rename=None,
) -> Dict[str, Any]:

    if splits_csv is not None:
        state[key_out] = pd.read_csv(splits_csv)
    elif not (cache and key_out in state):
        if not isinstance(reader_fn, list):
            reader_fn = [reader_fn]
        dfs = [fn(state=state) for fn in reader_fn]
        df = pd.concat(dfs).reset_index()
        state[key_out] = df
    else:
        logger.info('Caching dataset metadata from state')

    for f in filters:
        state[key_out] = f(state[key_out])

    if rename is not None:
        for r in rename:
            state[key_out][r['column']] = state[key_out][r['column']].apply(
                lambda x: r['new_value'] if x == r['value'] else x
            )

    return state


def load_voxceleb1_metadata(df_path: str):
    df = pd.read_csv(df_path, sep="\t")
    df.set_index('VoxCeleb1 ID', inplace=True)
    metadata = df.to_dict('index')
    return metadata


def read_audiodir(
    state,
    dataset_path,
    subsample=None,
    dataset=None,
    regex_groups=None,
    filter_list=None,
    partition_lists=None,
    filter_mode='include',
    dataset_metadata_csv=None,
    dataset_metadata_loader=load_voxceleb1_metadata,
):
    if not isinstance(dataset_path, list):
        dataset_path = [dataset_path]

    all_files = []

    for p in dataset_path:
        all_files_i = list(Path(p).rglob('*.wav')) + list(Path(p).rglob('*.flac'))
        all_files.extend(all_files_i)

    dataset_metadata: Optional[Dict[str, Any]] = None
    if dataset_metadata_csv is not None:
        dataset_metadata = dataset_metadata_loader(dataset_metadata_csv)
        state["speakers_metadata"] = dataset_metadata

    if filter_list is not None:
        with open(filter_list, 'r') as f:
            keep_values = set(f.read().splitlines())
        n_slashes = len(next(iter(keep_values)).split('/')) - 1
        stem_to_f = {'/'.join(v.parts[-n_slashes - 1 :]): v for v in all_files}
        if filter_mode == 'include':
            all_files = [stem_to_f[k] for k in keep_values]
        elif filter_mode == 'discard':
            all_files = [v for k, v in stem_to_f.items() if k not in keep_values]
        else:
            raise Exception("Unrecognized filter_mode {}".format(filter_mode))
    rows = []
    if subsample is not None:
        subsample_idx = np.random.choice(
            np.arange(len(all_files)), size=subsample, replace=False
        )
        all_files = np.array(all_files)[subsample_idx]
    for f in tqdm(all_files):
        try:
            finfo = sf.info(f)
            metadata = {
                'filename': str(f.resolve()),
                'sr': finfo.samplerate,
                'channels': finfo.channels,
                'frames': finfo.frames,
                'duration': finfo.duration,
            }

            if regex_groups is not None:
                regex_data = re.match(
                    regex_groups, str(f.relative_to(dataset_path[0]))
                ).groupdict()
                metadata.update(regex_data)

                if dataset_metadata is not None and 'speaker_id' in regex_data:
                    metadata.update(dataset_metadata[regex_data['speaker_id']])

            rows.append(metadata)
        except Exception as e:
            print(f'Failed reading {f}. {e}')
    df = pd.DataFrame(rows)
    if dataset is not None:
        df['dataset'] = dataset
    df['rel_path'] = df['filename'].apply(
        lambda x: str(Path(x).relative_to(dataset_path[0]))
    )
    if partition_lists is not None:
        remainder = None
        map_to_partitions = {}
        for k, v in partition_lists.items():
            if v is not None:
                list_path = Path(dataset_path[0], v)
                with open(list_path, 'r') as f:
                    list_files = f.read().splitlines()
                for l in list_files:
                    map_to_partitions[str(l)] = k
            else:
                remainder = k
        df['partition'] = df['rel_path'].apply(
            lambda x: map_to_partitions[x] if x in map_to_partitions else remainder
        )
        df = df.drop('rel_path', axis=1)
    return df


def subsample_dataset(
    state: Dict[str, Any],
    n_speakers: int,
    k_audios: int,
    max_length: float,
    out_key: str = "filtered_dataset_metadata",
) -> Dict[str, Any]:

    if out_key in state:
        logger.info('Caching dataset metadata from state')
        return state

    dataset_df = state["dataset_metadata"].copy()

    dataset_df = dataset_df[dataset_df["duration"] < max_length]

    # Get speakers that have more than k audios from different videos
    possible_speakers = [
        sid
        for sid in dataset_df['speaker_id'].unique()
        if len(dataset_df[dataset_df['speaker_id'] == sid]['video_id'].unique())
        > k_audios
    ]

    # Chose equally by gender
    male_speakers = [
        sid
        for sid in possible_speakers
        if state["speakers_metadata"][sid]["Gender"] == "m"
    ]
    female_speakers = [
        sid
        for sid in possible_speakers
        if state["speakers_metadata"][sid]["Gender"] == "f"
    ]

    n_speakers_by_gender = int(n_speakers / 2)
    chosen_male_speakers = sample(population=male_speakers, k=n_speakers_by_gender)
    chosen_female_speakers = sample(population=female_speakers, k=n_speakers_by_gender)

    chosen_speakers = chosen_male_speakers + chosen_female_speakers

    # Get audios from chosen speakers
    chosen_speakers_dfs = []
    for sid in chosen_speakers:
        speaker_df = dataset_df[dataset_df['speaker_id'] == sid]
        video_ids = speaker_df['video_id'].unique().tolist()
        chosen_videos = sample(population=video_ids, k=k_audios)

        audios_df = []
        for video_id in chosen_videos:
            video_df = speaker_df[speaker_df['video_id'] == video_id]
            chosen_audio = sample(population=video_df["segment_id"].tolist(), k=1)[0]
            audio_df = video_df[video_df["segment_id"] == chosen_audio]
            audios_df.append(audio_df)

        speaker_df = pd.concat(audios_df)
        chosen_speakers_dfs.append(speaker_df)

    subsample_df = pd.concat(chosen_speakers_dfs)
    state[out_key] = subsample_df

    return state


def subsample_dataset_with_fixed_n(
    state: Dict[str, Any],
    n_speakers: int,
    proportions: Dict[str, float],
    out_key: str = "filtered_dataset_metadata",
) -> Dict[str, Any]:
    """
    This method requires that the splits are created with the intended logic
    """

    if out_key in state:
        logger.info('Caching dataset metadata from state')
        return state

    dataset_df = state["dataset_metadata"].copy()

    first_n_speakers = dataset_df.speaker_id.unique()[:n_speakers]
    dataset_df = dataset_df[dataset_df.speaker_id.isin(first_n_speakers)]

    # Get audios from chosen speakers
    chosen_speakers_dfs = []
    for speaker_id in dataset_df.speaker_id.unique():
        speaker_df = dataset_df[dataset_df.speaker_id == speaker_id].copy()
        sample_sizes = process_sample_sizes(proportions=proportions, speaker_df=speaker_df)

        for split, sample_size in sample_sizes.items():
            speaker_split_df = speaker_df[speaker_df.set == split]
            if len(speaker_split_df) < sample_size:
                raise ValueError(f"Sample size bigger than the amount of audios for speaker {speaker_id}: set {split}, sample size {sample_size}.")
            chosen_speakers_dfs.append(speaker_split_df[:sample_size])

    subsample_df = pd.concat(chosen_speakers_dfs)
    state[out_key] = subsample_df

    return state