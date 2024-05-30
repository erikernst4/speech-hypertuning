import os
from typing import Any, Dict, Optional

import torch
import torchaudio
from s3prl.nn import S3PRLUpstream
from tqdm import tqdm


def save_upstream_embeddings(
    state: Dict[str, Any],
    saving_path: str,
    upstream: str = 'wavlm_base_plus',
    df_key: str = "filtered_dataset_metadata",
    cached_df: bool = True,
    cached_embeddings: bool = True,
    extract_embedding_method: Optional[callable] = None,
) -> Dict[str, Any]:
    extract_embedding_method = extract_embedding_method if extract_embedding_method is not None else extract_upstream_embedding 
    upstream = S3PRLUpstream(upstream)
    upstream.eval()

    os.makedirs(saving_path, exist_ok=True)

    # Get dataset audios filenames
    dataset_df = state[df_key]

    if cached_embeddings and cached_df and "embedding_filename" in dataset_df.columns:
        # Use cached filenames
        return state

    embeddings_paths = []
    for _, row in tqdm(dataset_df.iterrows()):
        embedding_saving_path = os.path.join(
            saving_path, f'{row["speaker_id"]}_{row["video_id"]}_{row["segment_id"]}.pt'
        )
        embeddings_paths.append(embedding_saving_path)
        if cached_embeddings and os.path.exists(embedding_saving_path):
            continue

        fname = row["filename"]
        waveform, _ = torchaudio.load(fname)

        # Obtain the valid length of the waveform
        valid_length = torch.tensor([waveform.size(1)])

        embeddings = extract_embedding_method(upstream, waveform, valid_length)

        torch.save(embeddings, embedding_saving_path)

    state[df_key]["embedding_filename"] = embeddings_paths
    return state


def extract_upstream_embedding_w_temporal_average_pooling(
    upstream: torch.nn.Module,
    wav: torch.Tensor,
    wav_len: torch.tensor,
) -> torch.Tensor:
    hidden_states, _ = upstream(wav, wav_len)

    # Concatenate hidden states from all layers
    all_layers_hidden_states = torch.cat(hidden_states)

    # Compute the average embedding across all layers
    average_embeddings = torch.mean(all_layers_hidden_states, dim=1)

    return average_embeddings


def extract_upstream_embedding_w_temporal_statistics_pooling(
    upstream: torch.nn.Module,
    wav: torch.Tensor,
    wav_len: torch.tensor,
) -> torch.Tensor:
    hidden_states, _ = upstream(wav, wav_len)

    # Concatenate hidden states from all layers
    all_layers_hidden_states = torch.cat(hidden_states)

    # Compute the average embedding across all layers
    average_embeddings = torch.mean(all_layers_hidden_states, dim=1)

    # Compute the std embedding across all layers
    std_embeddings = torch.std(all_layers_hidden_states, dim=1)

    pooled_embeddings = torch.cat([average_embeddings, std_embeddings], dim=-1)
    return pooled_embeddings

def extract_upstream_embedding_w_temporal_minmax_pooling(
    upstream: torch.nn.Module,
    wav: torch.Tensor,
    wav_len: torch.tensor,
) -> torch.Tensor:
    hidden_states, _ = upstream(wav, wav_len)

    # Concatenate hidden states from all layers
    all_layers_hidden_states = torch.cat(hidden_states)

    min_embeddings = torch.min(all_layers_hidden_states, dim=1).values
    max_embeddings = torch.max(all_layers_hidden_states, dim=1).values

    pooled_embeddings = torch.cat([min_embeddings, max_embeddings], dim=-1)
    return pooled_embeddings

def extract_upstream_embedding(
    upstream: torch.nn.Module,
    wav: torch.Tensor,
    wav_len: torch.tensor,
) -> torch.Tensor:
    hidden_states, _ = upstream(wav, wav_len)

    return torch.cat(hidden_states)  # list of embeddings to torch.Tensor
