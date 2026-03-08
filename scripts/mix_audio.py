"""
ESAS Audio Mixer for generating synthetic acoustic scenes with sound events.

This module implements the ESAS (Environmental Sound Augmentation System) audio
mixing pipeline, which combines background scenes from CochlScene with
foreground sound events from FSD50K to create synthetic audio clips for
acoustic scene classification and sound event detection tasks.

The mixing process ensures strict data separation between training,
validation, and test splits by using separate event databases (dev for
train/val, eval for test). It supports three mix types:
- background-only: no events, only scene audio.
- known-event: mix known events (from training set) into the scene.
- syth-unknown: mix unknown events (from test set) into the scene (test split only).

Events are placed with allowed overlap to create realistic acoustic scenarios,
and each event's audio can be time-stretched and pitch-shifted for variation.
The script also tracks file usage to ensure balanced sampling and prevent
excessive reuse of the same event audio files.

Usage:
    python mix_audio.py --split {train,val,test} --clips_per_scene N [--max_reuse_per_file M]
"""

import random
import json
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict, Counter
import re
from typing import List, Dict, Tuple, Set, Optional, Union


class ESASAudioMixer:
    """
    ESAS Audio Mixer for generating synthetic acoustic scenes with sound events.

    This class synthesizes audio clips by mixing background scenes from CochlScene
    with foreground sound events from FSD50K, following the ESAS dataset specification.
    The mixing process features enhanced audio file management to ensure diverse
    sampling of event sounds while maintaining strict data separation between splits.

    Attributes:
        scene_dir (Path): Directory containing CochlScene dataset.
        event_dir (Path): Directory containing FSD50K dataset.
        output_dir (Path): Output directory for generated clips.
        sr (int): Target sampling rate (Hz).
        target_duration (float): Target clip duration (seconds).
        known_snr_range (list): SNR range for known event mixing [min, max] (dB).
        unknown_snr_range (list): SNR range for unknown event mixing [min, max] (dB).
        max_event_types (int): Maximum number of event types per clip.
        max_same_event (int): Maximum repetitions of same event type.
        max_reuse_per_file (int): Maximum reuse count per event audio file.
        stretch_range (list): Time stretching range [min, max].
        pitch_shift_range (list): Pitch shifting range in semitones [min, max].
        event_list (pd.DataFrame): DataFrame containing event information.
        exclusion_list (dict): Dictionary of events to exclude per scene.
        scene_event_groupings (dict): Scene-specific allowed events loaded from JSON.
        dev_event_db (dict): Database of events from FSD50K dev split.
        eval_event_db (dict): Database of events from FSD50K eval split.
        scene_files_cache (dict): Cache of available scene files per scene and split.
        used_scene_files (set): Set of scene files already used.
        event_file_usage (defaultdict): Usage counter for each event audio file.
    """

    def __init__(self, config: dict):
        """
        Initialize the ESAS Audio Mixer with enhanced event management.

        Args:
            config (dict): Configuration dictionary containing:
                - scene_dir: Path to CochlScene dataset.
                - event_dir: Path to FSD50K dataset.
                - output_dir: Output directory for generated clips.
                - sampling_rate: Target sampling rate (Hz).
                - target_duration: Target clip duration (seconds).
                - known_snr_range: SNR range for known event mixing [min, max] (dB).
                - unknown_snr_range: SNR range for unknown event mixing [min, max] (dB).
                - max_event_types_per_clip: Maximum number of event types per clip.
                - max_same_event: Maximum repetitions of same event type.
                - max_reuse_per_file: Maximum reuse count per event audio file.
                - stretch_range: Time stretching range [min, max].
                - pitch_shift_range: Pitch shifting range in semitones [min, max].
                - event_list_path: Path to event list CSV.
                - exclusion_list_path: Path to exclusion list JSON.
                - scene_grouping_path: Path to scene-event grouping JSON (optional).
        """
        self.scene_dir = Path(config['scene_dir'])
        self.event_dir = Path(config['event_dir'])
        self.output_dir = Path(config['output_dir'])
        self.sr = config['sampling_rate']
        self.target_duration = config['target_duration']

        # Separate SNR RANGES for known and unknown events (no bias)
        self.known_snr_range = config.get('known_snr_range', [-15, 15])
        self.unknown_snr_range = config.get('unknown_snr_range', [-15, 15])

        # Validate parameters
        if self.known_snr_range[0] > self.known_snr_range[1]:
            raise ValueError(f"Invalid known SNR range: {self.known_snr_range}")
        if self.unknown_snr_range[0] > self.unknown_snr_range[1]:
            raise ValueError(f"Invalid unknown SNR range: {self.unknown_snr_range}")

        self.max_event_types = config['max_event_types_per_clip']
        self.max_same_event = config.get('max_same_event', 3)
        self.max_reuse_per_file = config.get('max_reuse_per_file', 2)
        self.stretch_range = config.get('stretch_range', [0.8, 1.15])
        self.pitch_shift_range = config.get('pitch_shift_range', [-3, 3])

        # Load event list and exclusion list
        self.event_list = self._load_event_list(config['event_list_path'])
        self.exclusion_list = self._load_json(config['exclusion_list_path'])

        # Load scene-event grouping JSON
        self.scene_grouping_path = config.get('scene_grouping_path', 'docs/event_scene_grouping.json')
        self.scene_event_groupings = self._load_json(self.scene_grouping_path).get('event_groupings', {})

        # Build SEPARATE event databases for different splits to prevent data leakage
        # dev_event_db: only for known events in train/val splits
        # eval_event_db: only for test split events (both known and unknown)
        self.dev_event_db = self._build_event_database('dev')
        self.eval_event_db = self._build_event_database('eval')

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize scene files cache and usage tracking
        self.scene_files_cache = {}
        self.used_scene_files = set()

        # Enhanced: audio file usage tracking for balanced sampling
        self.event_file_usage = defaultdict(Counter)  # {event_label: {file_path: usage_count}}

        print(f"ESAS Audio Mixer initialized for {self.output_dir}")
        print(f"Known events: {len(self.get_known_events())}")
        print(f"Unknown events: {len(self.get_unknown_events())}")
        print(f"\nSNR Configuration:")
        print(f"  Known SNR range: {self.known_snr_range[0]} to {self.known_snr_range[1]} dB (uniform)")
        print(f"  Unknown SNR range: {self.unknown_snr_range[0]} to {self.unknown_snr_range[1]} dB (uniform)")
        print(f"\nEnhanced Audio File Management:")
        print(f"  Maximum reuse per event file: {self.max_reuse_per_file}")
        print(f"  Balanced sampling across all available event audio files")
        print(f"  Usage tracking for optimal file distribution")
        print(f"\nStretch range: {self.stretch_range}")
        print(f"Pitch shift range: {self.pitch_shift_range}")
        print("\nData separation strategy:")
        print("- Train/Val: known events from dev split ONLY")
        print("- Test: known/unknown events from eval split ONLY")
        print("- Strict separation to prevent data leakage")
        print("\nEvent placement strategy:")
        print("- Both known and unknown events allow overlap")
        print("- More realistic and challenging acoustic scenarios")
        print("\nAudio saving:")
        print("- All clip types (background-only, known-event, syth-unknown) save audio files")
        print(f"\nScene-specific event grouping loaded from: {self.scene_grouping_path}")
        print(f"Number of scenes with predefined events: {len(self.scene_event_groupings)}")

    def _load_event_list(self, file_path: str) -> pd.DataFrame:
        """
        Load event list CSV file.

        Args:
            file_path (str): Path to event list CSV.

        Returns:
            pd.DataFrame: DataFrame containing event information with columns:
                - event_name: name of the event.
                - mix_type: 'known' or 'unknown'.

        Raises:
            ValueError: If file cannot be loaded.
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded event list: {len(df)} events")
            return df
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")

    def get_known_events(self) -> list:
        """
        Get list of known events from the event list.

        Returns:
            list: List of known event names.
        """
        known_events = self.event_list[self.event_list['mix_type'] == 'known']['event_name'].tolist()
        return known_events

    def get_unknown_events(self) -> list:
        """
        Get list of unknown events from the event list.

        Returns:
            list: List of unknown event names.
        """
        unknown_events = self.event_list[self.event_list['mix_type'] == 'unknown']['event_name'].tolist()
        return unknown_events

    def get_allowed_events_for_scene(self, scene_label: str, mix_type: str) -> list:
        """
        Retrieve the list of events allowed for a specific scene and mix type.

        Falls back to the global known/unknown event list if the scene is not defined
        or the specific event type list is missing.

        Args:
            scene_label (str): Scene class label.
            mix_type (str): Type of mix ('known-event' or 'syth-unknown').

        Returns:
            list: List of event names allowed for this scene and mix type.
        """
        # Default to global lists if scene grouping is unavailable
        if mix_type == 'known-event':
            default_list = self.get_known_events()
        else:  # syth-unknown
            default_list = self.get_unknown_events()

        # Check if scene exists in grouping
        scene_group = self.scene_event_groupings.get(scene_label)
        if scene_group is None:
            return default_list

        # Extract the appropriate event list from grouping
        if mix_type == 'known-event':
            allowed = scene_group.get('known_events', [])
        else:
            allowed = scene_group.get('unknown_events', [])

        # If the scene-specific list is empty, fallback to global
        if not allowed:
            return default_list
        return allowed

    def _build_event_database(self, dataset_type: str) -> dict:
        """
        Build event database for specific dataset type with duration metadata.

        CRITICAL: Build separate databases for dev and eval splits to prevent
        data leakage between train/val and test sets.

        Args:
            dataset_type (str): Type of dataset to build: 'dev' or 'eval'.

        Returns:
            dict: Dictionary mapping event labels to lists of (audio_file_path, duration) tuples.
        """
        event_db = defaultdict(list)

        # Determine which audio directory to scan based on dataset type
        if dataset_type == 'dev':
            audio_dir = self.event_dir / "FSD50K.dev_audio"
            metadata_path = self.event_dir / "FSD50K.ground_truth" / "dev.csv"
        elif dataset_type == 'eval':
            audio_dir = self.event_dir / "FSD50K.eval_audio"
            metadata_path = self.event_dir / "FSD50K.ground_truth" / "eval.csv"
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}. Must be 'dev' or 'eval'")

        if not audio_dir.exists():
            print(f"Warning: Audio directory not found: {audio_dir}")
            return dict(event_db)

        # Get all events from the event list
        all_events = self.get_known_events() + self.get_unknown_events()

        # Check for metadata file
        if metadata_path.exists():
            try:
                metadata_df = pd.read_csv(metadata_path)

                # Check for standard FSD50K format
                if 'fname' in metadata_df.columns and 'labels' in metadata_df.columns:
                    for idx, row in metadata_df.iterrows():
                        filename = row['fname']
                        labels_str = row['labels']

                        # Labels can be multiple (comma-separated)
                        labels = [label.strip() for label in labels_str.split(',')]

                        # Build full path
                        audio_path = audio_dir / f"{filename}.wav"

                        if audio_path.exists():
                            # Enhanced: Calculate and store duration for better balancing
                            try:
                                audio_info = sf.info(audio_path)
                                duration = audio_info.duration
                            except Exception as e:
                                print(f"Warning: Could not get duration for {audio_path}: {e}")
                                duration = 0

                            # Add to database for each relevant label
                            for label in labels:
                                if label in all_events:
                                    event_db[label].append((str(audio_path), duration))
                else:
                    print(f"Warning: Metadata file {metadata_path} has unexpected format")
                    return self._fallback_build_event_database(audio_dir, all_events)

            except Exception as e:
                print(f"Error parsing metadata {metadata_path}: {e}")
                return self._fallback_build_event_database(audio_dir, all_events)
        else:
            print(f"Warning: Metadata file not found: {metadata_path}")
            return self._fallback_build_event_database(audio_dir, all_events)

        # Print detailed statistics
        print(f"\nBuilt {dataset_type} event database:")
        total_files = sum(len(files) for files in event_db.values())
        print(f"  - Unique event types: {len(event_db)}")
        print(f"  - Total audio files: {total_files}")

        # Calculate and display average files per event
        if event_db:
            avg_files = total_files / len(event_db)
            print(f"  - Average files per event: {avg_files:.1f}")

            # Show file distribution for first few events
            print(f"  - Sample file distribution:")
            for event_label in sorted(event_db.keys())[:5]:  # Show first 5 for brevity
                file_count = len(event_db[event_label])
                if file_count > 0:
                    # Calculate average duration for this event
                    durations = [d for _, d in event_db[event_label] if d > 0]
                    avg_duration = np.mean(durations) if durations else 0
                    print(f"      {event_label}: {file_count} files, avg duration: {avg_duration:.1f}s")

            if len(event_db) > 5:
                print(f"      ... and {len(event_db) - 5} more event types")

        return dict(event_db)

    def _fallback_build_event_database(self, audio_dir: Path, all_events: list) -> dict:
        """
        Fallback method to build event database using filename parsing when metadata is unavailable.

        Args:
            audio_dir (Path): Directory containing audio files.
            all_events (list): List of all event names to include.

        Returns:
            dict: Event database with (file_path, duration) tuples.
        """
        event_db = defaultdict(list)

        # Helper function to normalize event names for matching
        def normalize_event_name(event_name):
            """Normalize event name for matching with filenames."""
            normalized = event_name.lower()
            normalized = re.sub(r'[^a-z0-9_]', '_', normalized)
            normalized = re.sub(r'_+', '_', normalized)
            return normalized

        # Create normalized event names mapping
        normalized_events = {normalize_event_name(event): event for event in all_events}

        # Scan audio directory
        wav_files = list(audio_dir.glob("*.wav"))

        for wav_file in wav_files:
            filename = wav_file.stem  # Remove extension

            # Normalize filename for matching
            normalized_filename = normalize_event_name(filename)

            # Try to find matching event
            label = None
            for norm_event, orig_event in normalized_events.items():
                if norm_event in normalized_filename or normalized_filename in norm_event:
                    label = orig_event
                    break

            if label and label in all_events:
                try:
                    audio_info = sf.info(wav_file)
                    duration = audio_info.duration
                except Exception as e:
                    print(f"Warning: Could not get duration for {wav_file}: {e}")
                    duration = 0

                event_db[label].append((str(wav_file), duration))

        print(f"Built fallback database from {audio_dir}:")
        print(f"  - Event types with files: {len(event_db)}")
        print(f"  - Total audio files: {sum(len(files) for files in event_db.values())}")

        return dict(event_db)

    def _load_json(self, file_path: str) -> dict:
        """
        Load JSON file with error handling.

        Args:
            file_path (str): Path to JSON file.

        Returns:
            dict: Loaded JSON data.

        Raises:
            ValueError: If file cannot be loaded.
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")

    def _load_audio(self, file_path: str, target_sr: int = None) -> Optional[np.ndarray]:
        """
        Load audio file with resampling.

        Args:
            file_path (str): Path to audio file.
            target_sr (int, optional): Target sampling rate. Defaults to self.sr.

        Returns:
            Optional[np.ndarray]: Loaded audio signal as a 1D numpy array, or None if loading fails.
        """
        if target_sr is None:
            target_sr = self.sr

        try:
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _get_available_scene_files(self, scene_label: str, split: str) -> list:
        """
        Get available scene files that haven't been used yet.

        Args:
            scene_label (str): Scene class label.
            split (str): Dataset split ('train', 'val', 'test').

        Returns:
            list: List of available scene file paths.
        """
        cache_key = f"{scene_label}_{split}"
        if cache_key not in self.scene_files_cache:
            split_path = self.scene_dir / split / scene_label
            if split_path.exists():
                scene_files = sorted(list(split_path.glob("*.wav")))
                available_files = []
                for scene_file in scene_files:
                    if str(scene_file) not in self.used_scene_files:
                        available_files.append(str(scene_file))

                self.scene_files_cache[cache_key] = available_files
            else:
                self.scene_files_cache[cache_key] = []

        return self.scene_files_cache[cache_key].copy()

    def _mark_scene_file_used(self, scene_file: str):
        """
        Mark a scene file as used to prevent reuse.

        Args:
            scene_file (str): Path to scene file that has been used.
        """
        self.used_scene_files.add(scene_file)
        for cache_key in self.scene_files_cache:
            if scene_file in self.scene_files_cache[cache_key]:
                self.scene_files_cache[cache_key].remove(scene_file)

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to prevent clipping (max amplitude scaled to 0.9 if exceeding 0.9).

        Args:
            audio (np.ndarray): Input audio signal.

        Returns:
            np.ndarray: Normalized audio signal.
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0.9:
            audio = audio * 0.9 / max_val
        return audio

    def _time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """
        Apply time stretching to audio.

        Args:
            audio (np.ndarray): Input audio signal.
            rate (float): Stretch rate (0.8 = 80% speed, 1.15 = 115% speed).

        Returns:
            np.ndarray: Time-stretched audio signal.
        """
        try:
            return librosa.effects.time_stretch(audio, rate=rate)
        except Exception as e:
            print(f"Error in time stretching: {e}")
            return audio

    def _pitch_shift(self, audio: np.ndarray, n_steps: float) -> np.ndarray:
        """
        Apply pitch shifting to audio.

        Args:
            audio (np.ndarray): Input audio signal.
            n_steps (float): Number of semitones to shift (-3 to +3).

        Returns:
            np.ndarray: Pitch-shifted audio signal.
        """
        try:
            return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
        except Exception as e:
            print(f"Error in pitch shifting: {e}")
            return audio

    def _adjust_length(self, audio: np.ndarray, target_length: float) -> np.ndarray:
        """
        Adjust audio length to target duration by either truncating or looping.

        Args:
            audio (np.ndarray): Input audio signal.
            target_length (float): Target duration in seconds.

        Returns:
            np.ndarray: Length-adjusted audio signal.
        """
        current_length = len(audio)
        target_samples = int(target_length * self.sr)

        if current_length < target_samples:
            repeats = int(np.ceil(target_samples / current_length))
            audio = np.tile(audio, repeats)
            audio = audio[:target_samples]
        elif current_length > target_samples:
            start = random.randint(0, current_length - target_samples)
            audio = audio[start:start + target_samples]

        return audio

    def _calculate_active_rms(self, audio: np.ndarray, threshold_db: float = -40) -> float:
        """
        Calculate RMS of active regions only (non-silent parts).

        Args:
            audio (np.ndarray): Input audio signal.
            threshold_db (float): Threshold in dB below which audio is considered silent.

        Returns:
            float: RMS value of active regions.
        """
        if len(audio) == 0:
            return 0.0

        # Convert threshold to linear scale
        threshold_linear = 10 ** (threshold_db / 20)

        # Find active regions (above threshold)
        audio_abs = np.abs(audio)
        active_mask = audio_abs > threshold_linear * np.max(audio_abs)

        if np.any(active_mask):
            active_audio = audio[active_mask]
            return np.sqrt(np.mean(active_audio ** 2))
        else:
            return 0.0

    def _calculate_gain_from_snr(self, scene_active_rms: float, event_active_rms: float, snr_db: float) -> float:
        """
        Calculate gain factor based on desired SNR using active region RMS.

        Args:
            scene_active_rms (float): RMS of background scene active region.
            event_active_rms (float): RMS of event audio active region.
            snr_db (float): Desired SNR in dB.

        Returns:
            float: Gain factor to apply to event audio.
        """
        if event_active_rms == 0:
            return 0.0

        snr_linear = 10 ** (snr_db / 10)
        target_event_rms = scene_active_rms / np.sqrt(snr_linear)
        gain = target_event_rms / event_active_rms
        gain *= 0.3  # Additional scaling factor for natural mixing
        return gain

    def _generate_timestamps_with_overlap(self, event_durations: list) -> tuple:
        """
        Generate timestamps for events with allowed overlap.

        Both known and unknown events can overlap to create more realistic and
        challenging acoustic scenarios. Overlap is allowed up to 50% of event duration.

        Args:
            event_durations (list): List of event durations in seconds.

        Returns:
            tuple: (start_times, sorted_durations) where start_times is a numpy array
                   and sorted_durations is the corresponding durations sorted by start time.
        """
        if not event_durations:
            return np.array([]), []

        start_times = []
        total_available_time = self.target_duration

        # Check if total event time is reasonable
        total_event_time = sum(event_durations)
        max_allowed_time = total_available_time * 2.5  # Allow 2.5x overlap density

        if total_event_time > max_allowed_time:
            # Scale down durations if there are too many events
            scale_factor = max_allowed_time / total_event_time
            event_durations = [d * scale_factor * 0.9 for d in event_durations]  # 0.9 for safety margin

        # Generate random start times for each event
        for i, duration in enumerate(event_durations):
            max_start = max(0, total_available_time - duration)

            # Determine placement strategy
            if i == 0:
                # First event: random placement
                start_time = random.uniform(0, max_start)
            else:
                # Decide whether to create overlap or separate placement
                # 50% chance to create overlap with previous events
                if random.random() < 0.5 and len(start_times) > 0:
                    # Create overlap with one of the previous events
                    reference_idx = random.randint(0, len(start_times) - 1)
                    reference_start = start_times[reference_idx]

                    # Overlap can range from -0.5*duration to +0.5*duration relative to reference
                    overlap_range = 0.5 * duration
                    min_start = max(0, reference_start - overlap_range)
                    max_start_for_overlap = min(max_start, reference_start + overlap_range)

                    if min_start < max_start_for_overlap:
                        start_time = random.uniform(min_start, max_start_for_overlap)
                    else:
                        # Fallback to random placement
                        start_time = random.uniform(0, max_start)
                else:
                    # Random placement without specific overlap intention
                    start_time = random.uniform(0, max_start)

            # Ensure event doesn't extend beyond clip duration
            if start_time + duration > total_available_time:
                start_time = total_available_time - duration

            start_times.append(start_time)

        # Sort by start time for natural progression
        sorted_indices = np.argsort(start_times)
        sorted_start_times = [start_times[i] for i in sorted_indices]
        sorted_durations = [event_durations[i] for i in sorted_indices]

        return np.array(sorted_start_times), sorted_durations

    def _get_excluded_events_for_scene(self, scene_label: str) -> list:
        """
        Get list of events to exclude for a specific scene based on exclusion list.

        Args:
            scene_label (str): Scene class label.

        Returns:
            list: List of event names to exclude.
        """
        excluded_events = []
        if scene_label in self.exclusion_list:
            excluded_events = self.exclusion_list[scene_label]
        return excluded_events

    def _get_available_event_files(self, event_label: str, event_db: dict,
                                   split: str, exclude_overused: bool = True) -> List[Tuple[str, float]]:
        """
        Get available event files considering usage history.

        Filters out overused files based on max_reuse_per_file parameter.

        Args:
            event_label (str): Event label.
            event_db (dict): Event database.
            split (str): Dataset split (unused, kept for consistency).
            exclude_overused (bool): Whether to exclude overused files.

        Returns:
            List[Tuple[str, float]]: List of (file_path, duration) tuples.
        """
        if event_label not in event_db:
            return []

        all_files = event_db[event_label].copy()

        if not exclude_overused:
            return all_files

        # Filter out overused files
        available_files = []
        for file_path, duration in all_files:
            usage_count = self.event_file_usage[event_label].get(file_path, 0)
            if usage_count < self.max_reuse_per_file:
                available_files.append((file_path, duration))

        return available_files

    def _adjust_target_event_count_based_on_availability(self, mix_type: str, scene_label: str,
                                                         event_db: dict, original_target: int,
                                                         split: str) -> int:
        """
        Dynamically adjust target event count based on available event files.

        This is particularly important for high event counts (e.g., 10 events) when
        the number of available audio files is limited. The function reduces the
        target if insufficient events with sufficient files exist.

        Args:
            mix_type (str): Type of mix ('known-event' or 'syth-unknown').
            scene_label (str): Scene class label.
            event_db (dict): Event database.
            original_target (int): Original target event count.
            split (str): Dataset split.

        Returns:
            int: Adjusted target event count.
        """
        if original_target <= 1:
            return original_target

        # Get available events for this scene
        available_events = self.get_allowed_events_for_scene(scene_label, mix_type)

        # Remove excluded events
        excluded_events = self._get_excluded_events_for_scene(scene_label)
        available_events = [e for e in available_events if e not in excluded_events]

        # Count available files per event
        available_files_per_event = []
        for event_label in available_events:
            available_files = self._get_available_event_files(event_label, event_db, split, exclude_overused=True)
            available_files_per_event.append((event_label, len(available_files)))

        # Sort by available files
        available_files_per_event.sort(key=lambda x: x[1], reverse=True)

        # For high event counts (10 events), check if we have enough unique events
        if original_target >= 10:
            # Count events with at least 2 available files (for diversity)
            events_with_sufficient_files = sum(1 for _, count in available_files_per_event if count >= 2)

            if events_with_sufficient_files < original_target:
                # Not enough events with sufficient files, reduce target
                adjusted_target = max(5, events_with_sufficient_files)  # Don't go below 5
                if adjusted_target < original_target:
                    print(f"  Adjusting {mix_type} target from {original_target} to {adjusted_target} "
                          f"(only {events_with_sufficient_files} events with sufficient files)")
                    return adjusted_target

        # For all event counts, check total available files
        total_available_files = sum(count for _, count in available_files_per_event)

        # Apply reduction factor for high targets when files are limited
        if original_target >= 5 and total_available_files < original_target * 2:
            # If files are limited, reduce target
            reduction_factor = max(0.5, total_available_files / (original_target * 2))
            adjusted_target = max(1, int(original_target * reduction_factor))

            if adjusted_target < original_target:
                print(f"  Adjusting {mix_type} target from {original_target} to {adjusted_target} "
                      f"(available files: {total_available_files})")
                return adjusted_target

        return original_target

    def _select_events_with_exact_count(self, mix_type: str, scene_label: str,
                                        event_db: dict, target_event_count: int,
                                        split: str) -> list:
        """
        Select events to achieve exact target event count with balanced audio file usage.

        Uses intelligent file selection to maximize diversity and minimize reuse of
        the same audio files. Ensures balanced sampling across all available event
        audio files.

        Args:
            mix_type (str): Type of mix ('known-event' or 'syth-unknown').
            scene_label (str): Scene class label.
            event_db (dict): Event database to select from.
            target_event_count (int): Exact number of events to select.
            split (str): Dataset split ('train', 'val', 'test').

        Returns:
            list: List of (event_label, event_file, estimated_duration) tuples.
        """
        selected_events = []

        if target_event_count == 0:
            return selected_events

        # Get available events for this scene
        available_events = self.get_allowed_events_for_scene(scene_label, mix_type)

        # Remove excluded events for this scene
        excluded_events = self._get_excluded_events_for_scene(scene_label)
        available_events = [e for e in available_events if e not in excluded_events]

        # Filter events that have available audio files (considering usage limits)
        available_events_with_files = []
        for event_label in available_events:
            available_files = self._get_available_event_files(event_label, event_db, split)
            if available_files:
                available_events_with_files.append((event_label, len(available_files)))

        if not available_events_with_files:
            print(f"  Warning: No available event files for scene={scene_label}, mix_type={mix_type} in {split} split")
            return selected_events

        # Sort events by available file count (use events with more files first)
        available_events_with_files.sort(key=lambda x: x[1], reverse=True)
        available_event_labels = [e[0] for e in available_events_with_files]

        # Strategy: distribute events evenly across available event types
        events_needed = target_event_count
        event_index = 0
        attempts = 0
        max_attempts = target_event_count * 10  # Prevent infinite loops

        while events_needed > 0 and available_event_labels and attempts < max_attempts:
            attempts += 1

            # Cycle through available event labels to ensure even distribution
            event_label = available_event_labels[event_index % len(available_event_labels)]

            # Get least used file for this event
            available_files = self._get_available_event_files(event_label, event_db, split)

            if available_files:
                # Sort by usage count (least used first)
                available_files_sorted = sorted(
                    available_files,
                    key=lambda x: self.event_file_usage[event_label].get(x[0], 0)
                )

                file_path, duration = available_files_sorted[0]

                # Update usage count
                self.event_file_usage[event_label][file_path] = \
                    self.event_file_usage[event_label].get(file_path, 0) + 1

                # Estimate duration after time stretching
                stretch_rate = random.uniform(self.stretch_range[0], self.stretch_range[1])
                estimated_duration = duration / stretch_rate if duration > 0 else 1.0
                estimated_duration = min(estimated_duration, 3.0)  # Max 3 seconds
                estimated_duration = max(estimated_duration, 0.5)  # Min 0.5 seconds

                selected_events.append((event_label, file_path, estimated_duration))
                events_needed -= 1
            else:
                # Remove event if no available files (all files overused)
                available_event_labels.remove(event_label)
                continue

            event_index += 1

        # If we still need more events but have exhausted all underused files
        if events_needed > 0:
            print(f"  Note: Could only select {target_event_count - events_needed} unique events "
                  f"for {target_event_count} requested (file usage limitation)")

            # Fallback: allow reuse of files even if over limit
            for event_label in available_events:
                if events_needed <= 0:
                    break

                all_files = event_db.get(event_label, [])
                if all_files:
                    # Use a file even if it's been used before (least used among all)
                    file_paths = [f[0] for f in all_files]
                    file_path = min(file_paths,
                                    key=lambda x: self.event_file_usage[event_label].get(x, 0))

                    # Find duration for this file
                    duration = next((d for f, d in all_files if f == file_path), 1.0)

                    # Estimate duration
                    stretch_rate = random.uniform(self.stretch_range[0], self.stretch_range[1])
                    estimated_duration = duration / stretch_rate if duration > 0 else 1.0
                    estimated_duration = min(estimated_duration, 3.0)
                    estimated_duration = max(estimated_duration, 0.5)

                    selected_events.append((event_label, file_path, estimated_duration))
                    events_needed -= 1

        # Randomize order for natural distribution
        random.shuffle(selected_events)

        # Log selection summary for monitoring
        if selected_events:
            event_counts = Counter([e[0] for e in selected_events])
        return selected_events

    def _get_snr_for_mix_type(self, mix_type: str) -> float:
        """
        Get appropriate SNR level based on mix type (uniform random sampling).

        Args:
            mix_type (str): Type of mix ('known-event' or 'syth-unknown').

        Returns:
            float: SNR level in dB (uniformly sampled from the corresponding range).
        """
        if mix_type == 'known-event':
            return random.uniform(self.known_snr_range[0], self.known_snr_range[1])
        else:  # syth-unknown
            return random.uniform(self.unknown_snr_range[0], self.unknown_snr_range[1])

    def _mix_events_into_scene(self, scene_audio: np.ndarray, events_to_mix: list,
                               scene_label: str, clip_snr_level: float, mix_type: str) -> tuple:
        """
        Mix multiple events into scene audio with allowed overlap for all event types.

        Args:
            scene_audio (np.ndarray): Background scene audio.
            events_to_mix (list): List of (event_label, event_file, estimated_duration) tuples.
            scene_label (str): Scene class label (unused, kept for consistency).
            clip_snr_level (float): Fixed SNR level for all events in this clip.
            mix_type (str): Type of mix (unused, kept for consistency).

        Returns:
            tuple: (mixed_audio, event_metadata) where event_metadata contains
                   timing and SNR information for each event.
        """
        mixed_audio = scene_audio.copy()
        event_metadata = []

        if events_to_mix:
            # Extract estimated durations
            event_durations = [event[2] for event in events_to_mix]

            # Generate timestamps with allowed overlap
            start_times, sorted_durations = self._generate_timestamps_with_overlap(event_durations)

            # Reorder events based on sorted timestamps
            sorted_events = []
            for i in range(len(start_times)):
                if i < len(events_to_mix):
                    sorted_events.append(events_to_mix[i])

            for idx, (event_label, event_file, estimated_duration) in enumerate(sorted_events):
                event_audio = self._load_audio(event_file)
                if event_audio is None:
                    continue

                # Apply time stretching for temporal variation
                stretch_rate = random.uniform(self.stretch_range[0], self.stretch_range[1])
                event_audio = self._time_stretch(event_audio, stretch_rate)

                # Apply pitch shifting for spectral variation
                pitch_shift = random.uniform(self.pitch_shift_range[0], self.pitch_shift_range[1])
                event_audio = self._pitch_shift(event_audio, pitch_shift)

                # Get actual duration after stretching
                actual_duration = len(event_audio) / self.sr
                event_duration = min(actual_duration, 3.0)  # Cap duration
                event_duration = max(event_duration, 0.5)  # Minimum duration

                # Get start time
                if idx < len(start_times):
                    start_time = start_times[idx]
                else:
                    # Fallback: distribute evenly
                    start_time = (idx / len(events_to_mix)) * (self.target_duration - event_duration)

                # Ensure event fits within audio duration
                max_possible_start = self.target_duration - event_duration
                if start_time > max_possible_start:
                    start_time = max_possible_start

                if start_time < 0:
                    start_time = 0
                    # Trim event if it still doesn't fit
                    event_duration = min(event_duration, self.target_duration)
                    event_audio = self._adjust_length(event_audio, event_duration)

                start_sample = int(start_time * self.sr)
                end_sample = start_sample + len(event_audio)

                if end_sample > len(mixed_audio):
                    # Trim event if it exceeds audio duration
                    event_audio = event_audio[:len(mixed_audio) - start_sample]
                    end_sample = len(mixed_audio)
                    event_duration = len(event_audio) / self.sr

                # Calculate RMS values for SNR-based mixing using active regions
                scene_segment = mixed_audio[start_sample:end_sample]
                scene_active_rms = self._calculate_active_rms(scene_segment)
                event_active_rms = self._calculate_active_rms(event_audio)

                # Calculate gain based on SNR using active region RMS
                gain = 0.0
                if event_active_rms > 0:
                    gain = self._calculate_gain_from_snr(scene_active_rms, event_active_rms, clip_snr_level)

                    # Apply fade-in and fade-out to avoid clicks
                    fade_duration = int(0.05 * self.sr)  # 50ms fade
                    if fade_duration > len(event_audio):
                        fade_duration = len(event_audio) // 2

                    if fade_duration > 0:
                        fade_in = np.linspace(0, 1, fade_duration)
                        fade_out = np.linspace(1, 0, fade_duration)
                        event_audio[:fade_duration] *= fade_in
                        if len(event_audio) > fade_duration:
                            event_audio[-fade_duration:] *= fade_out

                    # Mix event with calculated gain
                    mixed_audio[start_sample:end_sample] += event_audio * gain

                event_metadata.append({
                    'event_label': event_label,
                    'snr_db': clip_snr_level,
                    'start_time': start_time,
                    'end_time': start_time + event_duration,
                    'applied_gain': gain,
                    'scene_active_rms': scene_active_rms,
                    'event_active_rms': event_active_rms,
                    'event_duration': event_duration,
                    'stretch_rate': stretch_rate,
                    'pitch_shift': pitch_shift
                })

        return mixed_audio, event_metadata

    def generate_clip(self, scene_label: str, clip_id: str, split: str,
                      mix_type: str = None, scene_file: str = None,
                      target_event_count: int = None, snr_level: float = None) -> Optional[dict]:
        """
        Generate a single synthetic audio clip with strict data separation.

        Uses appropriate event databases based on split and mix type.
        Saves audio files for ALL clip types.

        Args:
            scene_label (str): Scene class label.
            clip_id (str): Unique clip identifier.
            split (str): Dataset split ('train', 'val', 'test').
            mix_type (str, optional): Specific mix type to generate
                ('background-only', 'known-event', 'syth-unknown').
            scene_file (str, optional): Specific scene file to use.
            target_event_count (int, optional): Exact number of events to mix.
            snr_level (float, optional): Fixed SNR level for this clip.

        Returns:
            Optional[dict]: Metadata dictionary for the generated clip, or None if generation fails.
        """
        if scene_file is None:
            raise ValueError("Scene file must be provided for unique background usage")

        # Check if scene file has been used
        if scene_file in self.used_scene_files:
            return None

        scene_audio = self._load_audio(scene_file)
        if scene_audio is None:
            return None

        # Adjust scene length and normalize
        scene_audio = self._adjust_length(scene_audio, self.target_duration)
        scene_audio = self._normalize_audio(scene_audio)

        # Select events with STRICT data separation
        events_to_mix = []
        if mix_type != 'background-only':
            # CRITICAL: Use appropriate event database based on split
            # Train/Val splits: only use dev_event_db (training data)
            # Test split: only use eval_event_db (evaluation data)

            if split in ['train', 'val']:
                # For train/val splits, only use dev split events
                event_db = self.dev_event_db

            else:  # test split
                # For test split, only use eval split events
                event_db = self.eval_event_db

            # Validate that we have events in the database
            if not event_db:
                print(f"  Warning: No events in {split} database for {mix_type}")
                return None

            if target_event_count is not None and target_event_count > 0:
                # NEW: Adjust target based on availability before selection
                adjusted_target = self._adjust_target_event_count_based_on_availability(
                    mix_type, scene_label, event_db, target_event_count, split
                )

                events_to_mix = self._select_events_with_exact_count(
                    mix_type, scene_label, event_db, adjusted_target, split
                )

        # Determine SNR level if not provided
        if snr_level is None and mix_type != 'background-only':
            snr_level = self._get_snr_for_mix_type(mix_type)

        # Process audio based on mix type
        if mix_type == 'background-only':
            # For background-only clips: just use the scene audio
            mixed_audio = scene_audio
            event_metadata = []
        else:
            # For event-mixed clips: mix events into scene
            mixed_audio, event_metadata = self._mix_events_into_scene(
                scene_audio, events_to_mix, scene_label, snr_level, mix_type
            )

        # Normalize final audio to prevent clipping
        mixed_audio = self._normalize_audio(mixed_audio)

        # Save audio file for ALL clip types (including background-only)
        output_filename = f"{clip_id}.wav"
        output_path = self.output_dir / output_filename
        sf.write(output_path, mixed_audio, self.sr)

        # Mark scene file as used (regardless of mix type)
        self._mark_scene_file_used(scene_file)

        # Prepare metadata (for all clip types)
        metadata = {
            'clip_id': clip_id,
            'original_scene': Path(scene_file).name,
            'scene_label': scene_label,
            'event_labels': [em['event_label'] for em in event_metadata],
            'event_snr': [em['snr_db'] for em in event_metadata],
            'event_timestamps': [[em['start_time'], em['end_time']] for em in event_metadata],
            'applied_gains': [em.get('applied_gain', 0.0) for em in event_metadata],
            'scene_active_rms': [em.get('scene_active_rms', 0.0) for em in event_metadata],
            'event_active_rms': [em.get('event_active_rms', 0.0) for em in event_metadata],
            'event_durations': [em.get('event_duration', 0.0) for em in event_metadata],
            'stretch_rates': [em.get('stretch_rate', 1.0) for em in event_metadata],
            'pitch_shifts': [em.get('pitch_shift', 0.0) for em in event_metadata],
            'split': split,
            'mix_type': mix_type if mix_type != 'background-only' else 'background-only',
            'duration': self.target_duration,
            'num_event_types': len(set([em['event_label'] for em in event_metadata])),
            'clip_snr_level': snr_level if snr_level is not None else 0.0
        }
        return metadata

    def check_event_file_availability(self, split: str):
        """
        Check event file availability before generation to anticipate limitations.

        Prints statistics about available files per event and warns about limited
        files for high event counts.

        Args:
            split (str): Dataset split to check.
        """
        print(f"\n=== Event File Availability Check for {split} split ===")

        if split in ['train', 'val']:
            event_db = self.dev_event_db
            event_types = self.get_known_events()
            print(f"Checking known events from dev split...")
        else:  # test
            event_db = self.eval_event_db
            known_events = self.get_known_events()
            unknown_events = self.get_unknown_events()
            event_types = known_events + unknown_events
            print(f"Checking both known and unknown events from eval split...")

        # Analyze each event type
        limited_events = []
        total_files = 0
        events_with_files = 0

        for event_label in sorted(event_types):
            if event_label in event_db:
                events_with_files += 1
                file_count = len(event_db[event_label])
                total_files += file_count

                if file_count > 0:
                    total_duration = sum(d for _, d in event_db[event_label] if d > 0)
                    avg_duration = total_duration / file_count if file_count > 0 else 0

                    # Estimate how many unique clips we can make
                    max_unique_clips = file_count * self.max_reuse_per_file

                    print(f"  {event_label}:")
                    print(f"    Files: {file_count}")
                    print(f"    Avg duration: {avg_duration:.1f}s")
                    print(f"    Max unique clips (reuse={self.max_reuse_per_file}): {max_unique_clips}")

                    # Warn if very limited for high event counts
                    if split in ['train', 'val'] and file_count < 20:
                        print(f"    ⚠ Limited files for Train/Val - may constrain 10-event clips")
                        limited_events.append((event_label, file_count))
                    elif split == 'test' and event_label in self.get_unknown_events() and file_count < 30:
                        print(f"    ⚠ Limited files for Test (unknown) - may constrain 10-event clips")
                        limited_events.append((event_label, file_count))

        print(f"\nOverall Statistics:")
        print(f"  Events with files: {events_with_files}/{len(event_types)}")
        print(f"  Total files available: {total_files}")
        if events_with_files > 0:
            avg_files_per_event = total_files / events_with_files
            print(f"  Average files per event: {avg_files_per_event:.1f}")

        if limited_events:
            print(f"\n⚠ WARNING: {len(limited_events)} events have limited files:")
            for event_label, file_count in limited_events[:10]:  # Show first 10
                print(f"    {event_label}: {file_count} files")
            if len(limited_events) > 10:
                print(f"    ... and {len(limited_events) - 10} more")

    def generate_dataset(self, split: str, clips_per_scene: int) -> list:
        """
        Generate dataset for specific split with strict data separation.

        The number of clips per scene is determined by the clips_per_scene parameter.
        The distribution among mix types (background-only, known-event, syth-unknown)
        follows predefined ratios per split:
            - train: 9:1 (background:known)
            - val:   1:1 (background:known)
            - test:  1:1:1 (background:known:unknown)
        Event counts (1,3,5,10) are distributed evenly across the clips.

        Args:
            split (str): Dataset split to generate ('train', 'val', 'test').
            clips_per_scene (int): Total number of clips to generate per scene (including all mix types).

        Returns:
            list: List of metadata dictionaries for all generated clips.
        """
        # First check event file availability for ALL splits
        self.check_event_file_availability(split)

        all_metadata = []
        clip_counter = 0

        # Get all scene labels from directory structure
        split_path = self.scene_dir / split
        scene_labels = [d.name for d in split_path.iterdir() if d.is_dir()]
        num_scenes = len(scene_labels)

        # Define distribution ratios and event counts per split
        if split == 'train':
            # Train: background-only : known-event = 9:1
            background_ratio = 0.9
            known_ratio = 0.1
            unknown_ratio = 0.0
            mix_types = ['background-only', 'known-event']
            known_event_counts = [1, 3, 5, 10]  # 1,3,5,10 events, equal distribution
            unknown_event_counts = []  # not used
        elif split == 'val':
            # Val: background-only : known-event = 1:1
            background_ratio = 0.5
            known_ratio = 0.5
            unknown_ratio = 0.0
            mix_types = ['background-only', 'known-event']
            known_event_counts = [1, 3, 5, 10]  # 1,3,5,10 events, equal distribution
            unknown_event_counts = []  # not used
        else:  # test
            # Test: background-only : known-event : syth-unknown = 1:1:1
            background_ratio = 1.0 / 3.0
            known_ratio = 1.0 / 3.0
            unknown_ratio = 1.0 / 3.0
            mix_types = ['background-only', 'known-event', 'syth-unknown']
            known_event_counts = [1, 3, 5, 10]  # 1,3,5,10 events for known-event
            unknown_event_counts = [1, 3, 5, 10]  # 1,3,5,10 events for syth-unknown

        print(f"\n{split.upper()} Distribution:")
        print(f"  Total scenes: {num_scenes}")
        print(f"  Clips per scene: {clips_per_scene}")
        print(f"  Background-only ratio: {background_ratio:.2f}")
        if known_ratio > 0:
            print(f"  Known-event ratio: {known_ratio:.2f} (event counts: {known_event_counts})")
        if unknown_ratio > 0:
            print(f"  Syth-unknown ratio: {unknown_ratio:.2f} (event counts: {unknown_event_counts})")

        print(f"\nSNR Strategy (uniform random):")
        print(f"  Known SNR range: {self.known_snr_range[0]} to {self.known_snr_range[1]} dB")
        if split == 'test':
            print(f"  Unknown SNR range: {self.unknown_snr_range[0]} to {self.unknown_snr_range[1]} dB")
        print(f"\nAudio File Management:")
        print(f"  Maximum reuse per file: {self.max_reuse_per_file}")
        print(f"  Balanced sampling across all available audio files")
        print(f"\nEvent placement: Overlap allowed for all event types")
        print(f"Audio saving: All clip types save audio files")

        for scene_label in scene_labels:
            print(f"\n{'=' * 60}")
            print(f"Processing scene: {scene_label}")
            print(f"{'=' * 60}")

            available_scene_files = self._get_available_scene_files(scene_label, split)
            if not available_scene_files:
                print(f"  No available background files for {scene_label}, skipping...")
                continue

            # Calculate number of clips needed for each mix type based on ratios
            total_needed = clips_per_scene
            clips_background = int(round(total_needed * background_ratio))
            clips_known = int(round(total_needed * known_ratio))
            clips_unknown = int(round(total_needed * unknown_ratio))

            # Adjust to ensure total matches exactly
            total_allocated = clips_background + clips_known + clips_unknown
            if total_allocated != total_needed:
                # Adjust the largest category to match
                diff = total_needed - total_allocated
                if clips_known >= clips_unknown and clips_known >= clips_background:
                    clips_known += diff
                elif clips_unknown >= clips_background:
                    clips_unknown += diff
                else:
                    clips_background += diff

            # For event mixes, further distribute across event counts
            clips_per_known_event_count = clips_known // len(known_event_counts) if known_event_counts else 0
            clips_per_unknown_event_count = clips_unknown // len(unknown_event_counts) if unknown_event_counts else 0

            # Adjust for remainder (simple: add remaining to first few event counts)
            remainder_known = clips_known - clips_per_known_event_count * len(known_event_counts)
            remainder_unknown = clips_unknown - clips_per_unknown_event_count * len(unknown_event_counts)

            print(f"\n  Using {len(available_scene_files)} background files, planning:")
            print(f"    Background-only: {clips_background} clips")
            if clips_known > 0:
                print(f"    Known-event: {clips_known} clips total ({clips_per_known_event_count} per base count, remainder {remainder_known})")
            if clips_unknown > 0:
                print(f"    Syth-unknown: {clips_unknown} clips total ({clips_per_unknown_event_count} per base count, remainder {remainder_unknown})")

            random.shuffle(available_scene_files)
            file_index = 0
            scene_clip_counter = 0

            # Generate background-only clips
            if clips_background > 0:
                print(f"\n  Generating background-only clips ({clips_background} total):")
                generated = 0
                for i in range(clips_background):
                    if file_index >= len(available_scene_files):
                        break
                    scene_file = available_scene_files[file_index]
                    clip_id = f"{scene_label}_{split}_{clip_counter:06d}"
                    metadata = self.generate_clip(
                        scene_label, clip_id, split, 'background-only', scene_file, 0, None
                    )
                    if metadata:
                        all_metadata.append(metadata)
                        clip_counter += 1
                        scene_clip_counter += 1
                        generated += 1
                        file_index += 1
                        if generated % 10 == 0:
                            print(f"    Generated {generated}/{clips_background} background-only clips")
                print(f"    Completed background-only: {generated}/{clips_background} clips")

            # Generate known-event clips
            if clips_known > 0 and known_event_counts:
                print(f"\n  Generating known-event clips ({clips_known} total):")
                generated = 0
                # Distribute across event counts
                for idx, event_count in enumerate(known_event_counts):
                    count_this = clips_per_known_event_count + (1 if idx < remainder_known else 0)
                    for _ in range(count_this):
                        if file_index >= len(available_scene_files) or generated >= clips_known:
                            break
                        scene_file = available_scene_files[file_index]
                        clip_id = f"{scene_label}_{split}_{clip_counter:06d}"
                        metadata = self.generate_clip(
                            scene_label, clip_id, split, 'known-event', scene_file, event_count, None
                        )
                        if metadata:
                            all_metadata.append(metadata)
                            clip_counter += 1
                            scene_clip_counter += 1
                            generated += 1
                            file_index += 1
                        if generated % 10 == 0:
                            print(f"    Generated {generated}/{clips_known} known-event clips")
                print(f"    Completed known-event: {generated}/{clips_known} clips")

            # Generate unknown-event clips (only for test)
            if clips_unknown > 0 and unknown_event_counts:
                print(f"\n  Generating syth-unknown clips ({clips_unknown} total):")
                generated = 0
                for idx, event_count in enumerate(unknown_event_counts):
                    count_this = clips_per_unknown_event_count + (1 if idx < remainder_unknown else 0)
                    for _ in range(count_this):
                        if file_index >= len(available_scene_files) or generated >= clips_unknown:
                            break
                        scene_file = available_scene_files[file_index]
                        clip_id = f"{scene_label}_{split}_{clip_counter:06d}"
                        metadata = self.generate_clip(
                            scene_label, clip_id, split, 'syth-unknown', scene_file, event_count, None
                        )
                        if metadata:
                            all_metadata.append(metadata)
                            clip_counter += 1
                            scene_clip_counter += 1
                            generated += 1
                            file_index += 1
                        if generated % 10 == 0:
                            print(f"    Generated {generated}/{clips_unknown} syth-unknown clips")
                print(f"    Completed syth-unknown: {generated}/{clips_unknown} clips")

            print(f"\n  Completed {scene_label}: {scene_clip_counter} clips generated")
            print(f"  Background files used: {file_index}/{len(available_scene_files)}")
            print(f"  Global clip counter: {clip_counter}")

        # Save metadata
        metadata_df = pd.DataFrame(all_metadata)
        metadata_path = f"data/metadata/{split}.csv"
        metadata_df.to_csv(metadata_path, index=False)

        self._print_generation_summary(all_metadata, split)
        return all_metadata

    def _print_generation_summary(self, all_metadata: list, split: str):
        """
        Print comprehensive generation summary including event file usage statistics.

        Args:
            all_metadata (list): List of all generated metadata.
            split (str): Dataset split.
        """
        print(f"\n{'=' * 80}")
        print(f"{split.upper()} DATASET GENERATION COMPLETE")
        print(f"{'=' * 80}")
        print(f"Total clips generated: {len(all_metadata)}")

        summary_df = pd.DataFrame(all_metadata)
        if not summary_df.empty:
            # Summary by mix type
            mix_summary = summary_df['mix_type'].value_counts()
            print(f"\nSummary by Mix Type:")
            for mix_type, count in mix_summary.items():
                percentage = (count / len(summary_df)) * 100
                print(f"  {mix_type}: {count} clips ({percentage:.1f}%)")

            # Summary by event counts with detailed distribution
            if 'num_event_types' in summary_df.columns:
                event_count_summary = summary_df.groupby(['mix_type', 'num_event_types']).size().unstack(fill_value=0)
                print(f"\nSummary by Number of Event Types:")
                print(event_count_summary)

            # Summary by SNR distribution at clip level
            if 'clip_snr_level' in summary_df.columns:
                print(f"\nSNR Distribution Analysis:")
                for mix_type in ['known-event', 'syth-unknown']:
                    if mix_type in summary_df['mix_type'].values:
                        snr_data = summary_df[(summary_df['mix_type'] == mix_type) &
                                              (summary_df['clip_snr_level'] > -100)]['clip_snr_level']
                        if not snr_data.empty:
                            print(f"\n{mix_type} SNR Statistics (uniform):")
                            print(f"  Mean: {snr_data.mean():.2f} dB")
                            print(f"  Std: {snr_data.std():.2f} dB")
                            print(f"  Min: {snr_data.min():.2f} dB")
                            print(f"  Max: {snr_data.max():.2f} dB")

            # Enhanced Event File Usage Analysis
            print(f"\n{'=' * 80}")
            print(f"ENHANCED EVENT FILE USAGE ANALYSIS")
            print(f"{'=' * 80}")

            if self.event_file_usage:
                total_files_used = 0
                total_usage_count = 0

                print(f"\nEvent File Usage Distribution:")
                for event_label, usage_counter in sorted(self.event_file_usage.items()):
                    if usage_counter:
                        files_used = len(usage_counter)
                        total_uses = sum(usage_counter.values())
                        avg_uses = total_uses / files_used if files_used > 0 else 0
                        usage_values = list(usage_counter.values())
                        min_uses = min(usage_values) if usage_values else 0
                        max_uses = max(usage_values) if usage_values else 0

                        print(f"\n  {event_label}:")
                        print(f"    Files used: {files_used}")
                        print(f"    Total uses: {total_uses}")
                        print(f"    Average uses per file: {avg_uses:.2f}")
                        print(f"    Min uses: {min_uses}")
                        print(f"    Max uses: {max_uses}")

                        total_files_used += files_used
                        total_usage_count += total_uses

                print(f"\nOverall Statistics:")
                print(f"  Total unique event files used: {total_files_used}")
                print(f"  Total event file uses: {total_usage_count}")

                if total_files_used > 0:
                    overall_avg = total_usage_count / total_files_used
                    print(f"  Overall average uses per file: {overall_avg:.2f}")

                    # Calculate how well we're using the available files
                    available_files_count = 0
                    used_files_count = 0
                    event_db = self.dev_event_db if split in ['train', 'val'] else self.eval_event_db

                    for event_label in self.event_file_usage.keys():
                        if event_label in event_db:
                            available_files_count += len(event_db[event_label])
                            used_files_count += len(self.event_file_usage[event_label])

                    if available_files_count > 0:
                        coverage_ratio = used_files_count / available_files_count
                        unused_files = available_files_count - used_files_count
                        print(f"  File coverage: {used_files_count}/{available_files_count} "
                              f"({coverage_ratio * 100:.1f}%)")
                        print(f"  Unused files: {unused_files}")

                        if coverage_ratio < 0.5:
                            print(f"  Warning: Low file coverage - consider increasing clip count or max_reuse_per_file")
                        elif coverage_ratio < 0.8:
                            print(f"  Moderate file coverage - acceptable for current settings")
                        else:
                            print(f"  ✓ Excellent file coverage achieved")

                        # Check for overused files
                        overused_files = 0
                        for event_label, usage_counter in self.event_file_usage.items():
                            for file_path, usage_count in usage_counter.items():
                                if usage_count > self.max_reuse_per_file:
                                    overused_files += 1
                        if overused_files > 0:
                            print(f"  Warning: {overused_files} files used more than {self.max_reuse_per_file} times")
                        else:
                            print(f"  ✓ All files within reuse limit ({self.max_reuse_per_file} times)")
            else:
                print(f"\nNo event files used in this split (background-only clips only)")

            print(f"\n{'=' * 80}")
            print(f"END OF FILE USAGE ANALYSIS")
            print(f"{'=' * 80}")

        print(f"\nMetadata saved to: data/metadata/{split}_metadata.csv")
        print(f"\n{'=' * 80}")
        print(f"DATASET GENERATION COMPLETE")
        print(f"{'=' * 80}")


def main():
    """Main execution function for ESAS audio mixing."""
    parser = argparse.ArgumentParser(description='ESAS Audio Mixing Script')
    parser.add_argument('--split', type=str, required=True,
                        choices=['train', 'val', 'test'],
                        help='Dataset split to generate')
    parser.add_argument('--clips_per_scene', type=int, required=True,
                        help='Number of clips to generate per scene (total for all mix types)')
    parser.add_argument('--max_reuse_per_file', type=int, default=2,
                        help='Maximum reuse count per event audio file (default: 2)')

    args = parser.parse_args()

    config = {
        'scene_dir': '.../CochlScene',
        'event_dir': '.../FSD50K',
        'output_dir': f'data/esas_data/{args.split}',
        'sampling_rate': 44100,
        'target_duration': 10.0,
        'known_snr_range': [-15, 15],
        'unknown_snr_range': [-15, 15],
        'max_event_types_per_clip': 10,
        'max_same_event': 3,
        'max_reuse_per_file': 2,
        'stretch_range': [0.8, 1.15],
        'pitch_shift_range': [-3, 3],
        'event_list_path': 'data/metadata/event_list.csv',
        'exclusion_list_path': 'docs/exclusion_list.json',
        'scene_grouping_path': 'docs/event_scene_grouping.json'
    }

    mixer = ESASAudioMixer(config)
    mixer.generate_dataset(split=args.split, clips_per_scene=args.clips_per_scene)


if __name__ == "__main__":
    main()