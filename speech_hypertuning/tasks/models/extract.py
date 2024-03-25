import os
from typing import Any, Dict

import torch
import torchaudio
from s3prl.nn import S3PRLUpstream
from tqdm import tqdm


def save_upstream_embeddings(
    state: Dict[str, Any],
    saving_path: str,
    upstream: str = 'wavlm_base_plus',
    df_key: str = "filtered_dataset_metadata",
    cached_df=True,
) -> Dict[str, Any]:
    upstream = S3PRLUpstream(upstream)

    os.makedirs(saving_path, exist_ok=True)

    # Get dataset audios filenames
    dataset_df = state[df_key]

    if cached_df and "embedding_filename" in dataset_df.columns:
        # Use cached filenames
        return state

    embeddings_paths = []
    for _, row in tqdm(dataset_df.iterrows()):
        embedding_saving_path = os.path.join(
            saving_path, f'{row["speaker_id"]}_{row["video_id"]}_{row["segment_id"]}.pt'
        )
        embeddings_paths.append(embedding_saving_path)
        if os.path.exists(embedding_saving_path):
            continue

        fname = row["filename"]
        waveform, _ = torchaudio.load(fname)

        # Obtain the valid length of the waveform
        valid_length = torch.tensor([waveform.size(1)])

        embedding = extract_upstream_embedding(upstream, waveform, valid_length)

        torch.save(embedding, embedding_saving_path)

    state[df_key]["embedding_filename"] = embeddings_paths
    return state


def extract_upstream_embedding(
    upstream: torch.nn.Module,
    wav: torch.Tensor,
    wav_len: torch.tensor,
) -> torch.Tensor:
    hidden_states, _ = upstream(wav, wav_len)

    # Concatenate hidden states from all layers
    all_layers_hidden_states = torch.cat(hidden_states)

    # Compute the average embedding across all layers
    average_embedding = torch.mean(all_layers_hidden_states, dim=0)

    return average_embedding
