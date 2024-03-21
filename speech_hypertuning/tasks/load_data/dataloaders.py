import copy

from torch.utils.data import Dataset


class DictDataset(Dataset):
    def __init__(
        self,
        metadata,
        state,
        out_cols,
        preprocessors=None,
        index_mapper=None,
        state_keys=None,
    ):
        self._metadata = metadata
        self._out_cols = out_cols
        self._state = {}
        self._state['metadata'] = metadata

        if state_keys is not None:
            for k in state_keys:
                if k in state:
                    self._state[k] = state[k]
        self._preprocessors = preprocessors
        if index_mapper is not None:
            self._idx_map = index_mapper(self._metadata)
        else:
            self._idx_map = list(range(len(self._metadata)))

    def __getitem__(self, idx):
        row = copy.deepcopy(self._metadata.iloc[self._idx_map[idx]])
        for p in self._preprocessors:
            row, self._state = p(row, self._state)
        out = {k: row[k] for k in self._out_cols}
        return out

    def __len__(self):
        return len(self._idx_map)
