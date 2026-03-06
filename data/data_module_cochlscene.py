import os
import torch
import librosa
import numpy as np
import lightning as L
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class AudioDataset(Dataset):
    """
    Dataset for CochlScene audio classification.

    Args:
        root_dir (str): Root directory containing data subsets.
        subset (str): Name of subset ('Train', 'Val', 'Test').
        clip_len (int): Clip length in seconds.
        sampling_rate (int): Sampling rate of waveforms.
    """

    def __init__(self, root_dir, subset, clip_len=10, sampling_rate=16000):
        self.root_dir = root_dir
        self.subset_dir = os.path.join(root_dir, subset)
        self.clip_len = clip_len
        self.sr = sampling_rate

        # Validate clip length
        assert 10 % clip_len == 0, "clip_len must divide 10 evenly"
        self.crops_per_clip = 10 // clip_len

        # Discover class folders
        self.scene_classes = sorted([
            d for d in os.listdir(self.subset_dir)
            if os.path.isdir(os.path.join(self.subset_dir, d))
        ])
        self.scene_to_idx = {s: i for i, s in enumerate(self.scene_classes)}

        # Gather file paths and labels
        self.filepaths = []
        self.labels = []
        for scene in self.scene_classes:
            scene_dir = os.path.join(self.subset_dir, scene)
            for f in os.listdir(scene_dir):
                if f.lower().endswith(".wav"):
                    self.filepaths.append(os.path.join(scene_dir, f))
                    self.labels.append(self.scene_to_idx[scene])

    def __len__(self):
        return len(self.filepaths) * self.crops_per_clip

    def _load_audio_file(self, path):
        """
        Load audio file using soundfile backend.

        Args:
            path (str): Path to audio file.

        Returns:
            tuple: (waveform, sample_rate)
        """
        # Primary method: soundfile
        try:
            wav_np, orig_sr = sf.read(path, dtype='float32')
            wav = torch.from_numpy(wav_np).float()

            # Convert multi-channel to mono
            if wav.dim() > 1:
                wav = wav.mean(dim=1)

            return wav, orig_sr

        except Exception as e:
            # Fallback: try torchaudio with soundfile backend
            try:
                wav, orig_sr = torchaudio.load(path, backend="soundfile")
                wav = wav.squeeze(0)  # Remove channel dimension
                return wav, orig_sr
            except:
                # Final fallback: librosa
                try:
                    wav_np, orig_sr = librosa.load(path, sr=None, mono=True)
                    wav = torch.from_numpy(wav_np).float()
                    return wav, orig_sr
                except ImportError:
                    raise RuntimeError(
                        f"Failed to load audio: {path}\n"
                        f"Soundfile error: {e}\n"
                        f"Install librosa as fallback: pip install librosa"
                    )

    def __getitem__(self, i):
        rec_idx = i // self.crops_per_clip
        crop_idx = i % self.crops_per_clip

        path = self.filepaths[rec_idx]
        label = torch.tensor(self.labels[rec_idx], dtype=torch.int64)

        # Load audio file
        wav, orig_sr = self._load_audio_file(path)

        # Resample if necessary
        if orig_sr != self.sr:
            wav = torchaudio.functional.resample(wav, orig_sr, self.sr)

        # Pad or trim to 10 seconds
        L = self.sr * 10
        current_length = wav.numel()

        if current_length < L:
            wav = torch.nn.functional.pad(wav, (0, L - current_length))
        elif current_length > L:
            wav = wav[:L]

        # Extract crop segment
        start = crop_idx * self.clip_len * self.sr
        end = start + self.clip_len * self.sr
        wav_crop = wav[start:end]

        return wav_crop, label


class AudioLabelsDataset(AudioDataset):
    """
    Dataset containing tuples of audio waveform, scene label, device label, city label, and event label (optional).

    Args:
        root_dir (str): Directory of dataset.
        subset (str): Name of required meta file. e.g. ``train``, ``valid``, ``test``...
        clip_len (int): Clip length.
        sampling_rate (int): Sampling rate of waveforms.
        event_tag_dir (str): Directory of .pt file containing soft event labels.
    """
    def __init__(self, root_dir: str, subset: str,
                 sampling_rate: int = 16000,
                 clip_len: int = 10,
                 event_tag_dir: str = None):
        super().__init__(root_dir, subset, clip_len, sampling_rate)
        if event_tag_dir:
            self.event_tag_dir = f"{event_tag_dir}_{subset.lower()}.pt"
            self.event_tags = torch.load(self.event_tag_dir).float()

        else:
            self.event_tag_dir = event_tag_dir

    def __getitem__(self, i):
        # Get the filename
        wav, label = super().__getitem__(i)
        data_dict = {
            'wav': wav,
            'scene': label,
        }

        if self.event_tag_dir:
            data_dict['event'] = self.event_tags[i]
        return data_dict


class CochlSceneDataModule(L.LightningDataModule):
    """
    CochlScene DataModule wrapping train, validation, test and predict DataLoaders.

    Args:
        root_dir (str): Directory of dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of workers to use for DataLoaders. Will save time for loading data to GPU but increase CPU usage.
        pin_memory (bool): If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them. Will save time for data loading.
    """
    def __init__(self,
                 root_dir: str,
                 batch_size: int = 16,
                 num_workers: int = 0,
                 pin_memory: bool=False,
                 train_subset="Train",
                 valid_subset="Val",
                 test_subset="Test",
                 predict_subset="Test",
                 **kwargs):
        super().__init__()
        self.root_dir = root_dir
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
            self.train_set = AudioLabelsDataset(self.root_dir, subset=self.train_subset, **self.kwargs)
            self.valid_set = AudioLabelsDataset(self.root_dir, subset=self.valid_subset, **self.kwargs)
        if stage == "validate":
            self.valid_set = AudioLabelsDataset(self.root_dir, subset=self.valid_subset, **self.kwargs)
        if stage == "test":
            self.test_set = AudioLabelsDataset(self.root_dir, subset=self.test_subset, **self.kwargs)
        if stage == "predict":
            self.predict_set = AudioDataset(self.root_dir, subset=self.predict_subset, **self.kwargs)

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
    data_dict = {}
    all_keys = {k for d in list_data_dict for k in d.keys()}  # union of all keys

    # Collect keys from first sample
    for key in all_keys:
        values = [d.get(key, None) for d in list_data_dict]

        if all(torch.is_tensor(v) for v in values):
            # Stack tensors into batch dimension
            data_dict[key] = torch.stack(values)
        else:
            # Keep as list (labels, integers, etc.)
            data_dict[key] = values
    return data_dict