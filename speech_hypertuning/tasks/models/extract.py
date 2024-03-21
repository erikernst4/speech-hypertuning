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
) -> None:
    upstream = S3PRLUpstream(upstream)

    # Get dataset audios filenames
    dataset_df = state[df_key]

    for _, row in tqdm(dataset_df.iterrows()):
        fname = row["filename"]
        waveform, _ = torchaudio.load(fname)
        waveform = waveform.unsqueeze(0)  # Add batch dimension

        # Obtain the valid length of the waveform
        valid_length = torch.tensor([waveform.size(1)])

        embedding = extract_upstream_embedding(upstream, waveform, valid_length)

        embedding_saving_path = os.path.join(
            saving_path, row["speaker_id"], row["video_id"], row["segment_id"]
        )
        torch.save(embedding, embedding_saving_path)


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
