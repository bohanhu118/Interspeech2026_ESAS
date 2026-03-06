import os
import random
import pandas as pd
import torch
import numpy as np
import lightning as L
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning.pytorch as pl


class AudioDataset(Dataset):
    """
    Dataset containing pairs of audio waveform and filename.

    Args:
        audio_dir (str): Directory of dataset.
        meta_dir (str): Directory of meta file.
        subset (str): Name of subset. e.g. ``Train``, ``Val``, ``Test``...
        clip_len (int): Clip length in seconds.
        sampling_rate (int): Sampling rate of waveforms.
    """

    def __init__(self, audio_dir, meta_dir, subset, clip_len=10, sampling_rate=16000):
        self.audio_dir = audio_dir
        self.subset = subset
        self.clip_len = clip_len
        self.sr = sampling_rate

        # Load metadata
        meta_path = os.path.join(meta_dir, f"{subset.lower()}.csv")
        self.meta = pd.read_csv(meta_path)

        # Filter out entries where audio_saved is False
        if 'audio_saved' in self.meta.columns:
            self.meta = self.meta[self.meta['audio_saved'] == True]
            self.meta = self.meta.reset_index(drop=True)

        # Validate clip length
        assert 10 % clip_len == 0, "clip_len must divide 10 evenly"
        self.crops_per_clip = 10 // clip_len

    def __len__(self):
        return len(self.meta) * self.crops_per_clip

    def _build_audio_path(self, clip_id):
        """Construct the correct audio file path based on clip_id."""
        if not clip_id.endswith('.wav'):
            clip_id = clip_id + '.wav'

        # Primary path: audio_dir/subset/clip_id.wav
        audio_path = os.path.join(self.audio_dir, self.subset.lower(), clip_id)

        if os.path.exists(audio_path):
            return audio_path

        # Fallback path: audio_dir/clip_id.wav
        fallback_path = os.path.join(self.audio_dir, clip_id)
        if os.path.exists(fallback_path):
            return fallback_path

        raise FileNotFoundError(f"Audio file not found: {clip_id}")

    def _load_audio_file(self, path):
        """Load audio file with primary soundfile backend."""

        wav_np, orig_sr = sf.read(path, dtype='float32')
        wav = torch.from_numpy(wav_np).float()

        # Convert multi-channel to mono
        if wav.dim() > 1:
            wav = wav.mean(dim=1)

        return wav, orig_sr

    def __getitem__(self, i):
        rec_idx = i // self.crops_per_clip
        crop_idx = i % self.crops_per_clip

        row_i = self.meta.iloc[rec_idx]
        clip_id = row_i['clip_id']

        # Build path and load audio
        path = self._build_audio_path(clip_id)
        wav, orig_sr = self._load_audio_file(path)

        # Resample if necessary
        if orig_sr != self.sr:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sr)
            wav = resampler(wav)

        # Ensure correct shape and length
        wav = wav.squeeze()
        L = self.sr * 10

        if wav.numel() < L:
            wav = torch.nn.functional.pad(wav, (0, L - wav.numel()))
        elif wav.numel() > L:
            wav = wav[:L]

        # Extract crop segment
        start = crop_idx * self.clip_len * self.sr
        end = start + self.clip_len * self.sr
        wav_crop = wav[start:end]

        return wav_crop


class AudioLabelsDataset(AudioDataset):
    """
    Extended Dataset returning waveform + various labels.

    Returns a dict containing:
        {
            'wav': waveform crop,
            'scene': scene_label,
            'domain': domain_label (optional),
            'event': event_label (optional)
        }
    """
    def __init__(self, event_tag_dir: str = None, domain: str = None, **kwargs):
        """
        Args:
            event_tag_dir (str): Directory containing <subset>.pt files of event tags.
            domain (str): Column name in meta specifying the domain label.
            **kwargs: Arguments forwarded to AudioDataset.
        """
        super().__init__(**kwargs)
        self.domain = domain
        # List scene classes and create label mapping
        self.scene_classes = sorted(self.meta['scene_label'].unique().tolist())
        self.scene_to_idx = {name: i for i, name in enumerate(self.scene_classes)}

        # List scene classes and create label mapping
        if self.domain:
            if self.domain not in self.meta.columns:
                raise RuntimeError(
                    f"Invalid domain column '{self.domain}'. "
                    f"Available columns: {list(self.meta.columns)}"
                )

            # Convert domain labels to strings to ensure consistent indexing
            domain_values = self.meta[self.domain].astype(str).unique().tolist()
            self.domain_classes = sorted(domain_values)
            self.domain_to_idx = {name: i for i, name in enumerate(self.domain_classes)}

        if event_tag_dir:
            tag_path = f"{event_tag_dir}_{self.subset.lower()}.pt"
            self.event_tags = torch.load(tag_path).float()
        else:
            self.event_tags = None

    def __getitem__(self, i):
        # Get the filename
        wav = super().__getitem__(i)
        rec_idx = i // self.crops_per_clip
        row_i = self.meta.iloc[rec_idx]
        scene_label = self.scene_to_idx[row_i['scene_label']]
        scene_label = torch.tensor(scene_label, dtype=torch.int64)
        data_dict = {
            'wav': wav,
            'scene': scene_label
        }

        if self.domain:
            domain_label = self.domain_to_idx[str(row_i[self.domain])]
            domain_label = torch.tensor(domain_label, dtype=torch.int64)
            data_dict['domain'] = domain_label

        if self.event_tags is not None:
            data_dict['event'] = self.event_tags[i]
        return data_dict


class ESASDataModule(L.LightningDataModule):
    """
    ESAS DataModule wrapping train, validation, test and predict DataLoaders.

    Args:
        audio_dir (str): Directory of dataset.
        meta_dir (str): Directory of meta file.
        batch_size (int): Batch size.
        num_workers (int): Number of workers to use for DataLoaders. Will save time for loading data to GPU but increase CPU usage.
        pin_memory (bool): If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them. Will save time for data loading.
    """
    def __init__(self,
                 audio_dir: str,
                 meta_dir: str,
                 batch_size: int = 16,
                 num_workers: int = 0,
                 pin_memory: bool=False,
                 train_subset="Train",
                 valid_subset="Val",
                 test_subset="Test",
                 predict_subset="Test",
                 **kwargs):
        super().__init__()
        self.audio_dir = audio_dir
        self.meta_dir = meta_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_subset = train_subset
        self.valid_subset = valid_subset
        self.test_subset = test_subset
        self.predict_subset = predict_subset
        self.collect_f = collate_fn
        self.kwargs = kwargs

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_set = AudioLabelsDataset(audio_dir=self.audio_dir, meta_dir=self.meta_dir, subset=self.train_subset, **self.kwargs)
            self.valid_set = AudioLabelsDataset(audio_dir=self.audio_dir, meta_dir=self.meta_dir, subset=self.valid_subset, **self.kwargs)
        if stage == "validate":
            self.valid_set = AudioLabelsDataset(audio_dir=self.audio_dir, meta_dir=self.meta_dir, subset=self.valid_subset, **self.kwargs)
        if stage == "test":
            self.test_set = AudioLabelsDataset(audio_dir=self.audio_dir, meta_dir=self.meta_dir, subset=self.test_subset, **self.kwargs)
        if stage == "predict":
            self.predict_set = AudioDataset(audio_dir=self.audio_dir, meta_dir=self.meta_dir, subset=self.predict_subset, **self.kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          pin_memory=self.pin_memory, collate_fn=self.collect_f)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory, collate_fn=self.collect_f)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory, collate_fn=self.collect_f)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory, collate_fn=self.collect_f)


def collate_fn(list_data_dict):
    """
    Collate function that handles nested dictionaries and string data like mix_type.

    Args:
        list_data_dict: List of dictionaries to collate

    Returns:
        Collated dictionary with stacked tensors
    """
    if not list_data_dict:
        return {}

    # Handle the case where list_data_dict might contain nested dictionaries
    if isinstance(list_data_dict[0], dict) and 'source' in list_data_dict[0]:
        # This is already a source-target batch, use the specialized collate function
        return collate_fn_with_target(list_data_dict)

    # Original collate logic for regular batches
    data_dict = {}
    all_keys = {k for d in list_data_dict for k in d.keys()}  # union of all keys

    # Collect keys from first sample
    for key in all_keys:
        values = [d.get(key, None) for d in list_data_dict]

        if all(torch.is_tensor(v) for v in values):
            # Stack tensors into batch dimension
            data_dict[key] = torch.stack(values)
        elif all(isinstance(v, (list, np.ndarray)) for v in values if v is not None):
            # Handle lists and numpy arrays
            non_none_values = [v for v in values if v is not None]
            if non_none_values:
                # Try to convert to tensor
                try:
                    data_dict[key] = torch.tensor(non_none_values)
                except:
                    # Keep as list if conversion fails
                    data_dict[key] = non_none_values
        elif all(isinstance(v, str) for v in values if v is not None):
            # Handle string data (like mix_type)
            data_dict[key] = values
        else:
            # Keep as list (labels, integers, strings, etc.)
            data_dict[key] = values

    return data_dict
