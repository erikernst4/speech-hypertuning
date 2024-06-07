import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from s3prl.nn import S3PRLUpstream
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def save_upstream_embeddings(
    state: Dict[str, Any],
    saving_path: str,
    upstream_name: str = 'wavlm_base_plus',
    df_key: str = "filtered_dataset_metadata",
    cached_df: bool = True,
    cached_embeddings: bool = True,
    extract_embedding_method: Optional[Callable] = None,
) -> Dict[str, Any]:
    extract_embedding_method = (
        extract_embedding_method
        if extract_embedding_method is not None
        else extract_upstream_embedding
    )
    max_length = 30  # Limit audio duration for inference
    upstream = S3PRLUpstream(upstream_name)
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

        # Load audio
        fname = row["filename"]
        waveform, _ = torchaudio.load(fname)
        # Chunk if necessary
        audio_info = sf.info(fname)
        desired_frames = int(max_length * audio_info.samplerate)
        total_frames = audio_info.frames
        if total_frames > desired_frames:
            waveform = waveform[0][:desired_frames].unsqueeze(dim=0)

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


def extract_upstream_embedding_w_std_pooling(
    upstream: torch.nn.Module,
    wav: torch.Tensor,
    wav_len: torch.tensor,
) -> torch.Tensor:
    hidden_states, _ = upstream(wav, wav_len)

    # Concatenate hidden states from all layers
    all_layers_hidden_states = torch.cat(hidden_states)

    # Compute the std embedding across all layers
    std_embeddings = torch.std(all_layers_hidden_states, dim=1)
    return std_embeddings


def extract_upstream_embedding_w_temporal_max_pooling(
    upstream: torch.nn.Module,
    wav: torch.Tensor,
    wav_len: torch.tensor,
) -> torch.Tensor:
    hidden_states, _ = upstream(wav, wav_len)

    # Concatenate hidden states from all layers
    all_layers_hidden_states = torch.cat(hidden_states)

    max_embeddings = torch.max(all_layers_hidden_states, dim=1).values

    return max_embeddings


def extract_upstream_embedding_w_temporal_min_pooling(
    upstream: torch.nn.Module,
    wav: torch.Tensor,
    wav_len: torch.tensor,
) -> torch.Tensor:
    hidden_states, _ = upstream(wav, wav_len)

    # Concatenate hidden states from all layers
    all_layers_hidden_states = torch.cat(hidden_states)

    min_embeddings = torch.min(all_layers_hidden_states, dim=1).values

    return min_embeddings


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


def extract_upstream_embedding_w_temporal_statistics_plus_pooling(
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

    min_embeddings = torch.min(all_layers_hidden_states, dim=1).values
    max_embeddings = torch.max(all_layers_hidden_states, dim=1).values

    pooled_embeddings = torch.cat(
        [average_embeddings, std_embeddings, min_embeddings, max_embeddings], dim=-1
    )
    return pooled_embeddings


def extract_upstream_embedding(
    upstream: torch.nn.Module,
    wav: torch.Tensor,
    wav_len: torch.tensor,
) -> torch.Tensor:
    hidden_states, _ = upstream(wav, wav_len)

    return torch.cat(hidden_states)  # list of embeddings to torch.Tensor


def calculate_dataset_upstream_mean_and_std(
    state,
    train_dataloader,
    saving_path: str,
    upstream_name: str = 'wavlm_base_plus',
    cached: bool = True,
):
    def get_embeddings_wo_padding(
        xs: torch.Tensor, xs_len: torch.LongTensor
    ) -> np.ndarray:
        """
        Flatten embeddings and remove padding
        """
        embedding_list = []
        for x in xs:
            for layer_idx, layer in enumerate(x):
                cut_padding_idx = xs_len[layer_idx]
                layer_embedding_wo_padding = layer[:cut_padding_idx]
                embedding_list.append(layer_embedding_wo_padding)
        return torch.cat(embedding_list).numpy()

    mean_saving_path = os.path.join(saving_path, f'dataset_mean.pt')
    std_saving_path = os.path.join(saving_path, f'dataset_std.pt')

    if cached and (
        os.path.exists(mean_saving_path) and os.path.exists(std_saving_path)
    ):
        state['dataset_mean'] = torch.load(mean_saving_path, weights_only=True)
        state['dataset_std'] = torch.load(std_saving_path, weights_only=True)
        return state

    upstream = S3PRLUpstream(upstream_name)
    upstream.eval()

    scaler = StandardScaler(with_mean=True, with_std=True)

    train_dataloader = state['dataloaders']['train']
    for batch in tqdm(train_dataloader):
        with torch.no_grad():
            hidden, hidden_lens = upstream(batch['wav'], wavs_len=batch['wav_lens'])
            # Out to tensor
            hidden = torch.stack(hidden).transpose(0, 1)
            hidden_lens = hidden_lens[
                0
            ]  # Before it's a list of length=upstream_layers, all elements are equal

            batch_embeddings = get_embeddings_wo_padding(hidden, hidden_lens)

            scaler.partial_fit(batch_embeddings)

    dataset_mean = torch.from_numpy(scaler.mean_)
    dataset_std = torch.from_numpy(np.sqrt(scaler.var_))

    state['dataset_mean'] = dataset_mean.to("cuda:0")
    state['dataset_std'] = dataset_std.to("cuda:0")

    torch.save(dataset_mean, mean_saving_path)
    torch.save(dataset_std, std_saving_path)

    return state


def calculate_dataset_pooled_mean_and_std(
    state,
    train_dataloader,
    saving_path: str,
    upstream_name: str = 'wavlm_base_plus',
    cached: bool = True,
):
    mean_saving_path = os.path.join(saving_path, f'dataset_mean.pt')
    std_saving_path = os.path.join(saving_path, f'dataset_std.pt')

    if cached and (
        os.path.exists(mean_saving_path) and os.path.exists(std_saving_path)
    ):
        state['dataset_mean'] = torch.load(mean_saving_path, weights_only=True)
        state['dataset_std'] = torch.load(std_saving_path, weights_only=True)
        return state

    upstream = S3PRLUpstream(upstream_name)
    upstream.eval()

    scaler = StandardScaler(with_mean=True, with_std=True)

    train_dataloader = state['dataloaders']['train']
    for batch in tqdm(train_dataloader):
        batch_embeddings = batch['upstream_embedding'].detach().numpy()
        batch_size, upstream_layers, embedding_dim = batch_embeddings.shape
        batch_embeddings = batch_embeddings.reshape(
            batch_size * upstream_layers, embedding_dim
        )
        scaler.partial_fit(batch_embeddings)

    dataset_mean = torch.from_numpy(scaler.mean_)
    dataset_std = torch.from_numpy(np.sqrt(scaler.var_))

    state['dataset_mean'] = dataset_mean.to("cuda:0")
    state['dataset_std'] = dataset_std.to("cuda:0")

    torch.save(dataset_mean, mean_saving_path)
    torch.save(dataset_std, std_saving_path)

    return state
