"""
ESAS Dataset Metadata Generator

This module provides tools for generating metadata for the Event-Shifted Acoustic Scene (ESAS) dataset.
It integrates CochlScene scene recordings with FSD50K sound event labels, identifies real-unknown
recordings (containing unknown events with high probability), and produces the necessary metadata files
for dataset construction and evaluation.

The generator performs the following tasks:
- Loads CochlScene metadata including BEATs event probabilities.
- Loads FSD50K ground truth and vocabulary for label mapping.
- Maps BEATs event MID to FSD50K labels.
- Identifies recordings that contain scene-specific unknown events (probability >= threshold) and marks them as "real-unknown".
- Produces event splits (known/unknown), metadata schema, scene-event coverage statistics, and CSV templates.

Usage:
    Run the script directly to generate all metadata components.
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


class ESASMetadataGenerator:
    """
    ESAS Dataset Metadata Generator

    This class generates metadata for the Event-Shifted Acoustic Scene (ESAS) dataset,
    including event splits, metadata schema, and scene-event coverage statistics.
    It also identifies real-unknown recordings from CochlScene based on BEATs probabilities
    and the scene-specific unknown event lists defined in the event-scene grouping JSON.

    Attributes:
        scene_dir (Path): Path to CochlScene dataset directory.
        event_mapping (dict): Loaded event-scene grouping dictionary.
        fsd50k_meta_dir (Path): Path to FSD50K metadata directory.
        scene_classes (list): List of scene labels from the event mapping.
        probability_threshold (float): Threshold above which an event is considered present.
        fsd50k_labels (list): All unique event labels from FSD50K.
        fsd50k_vocabulary (dict): Mapping from MID to FSD50K label.
        beats_to_fsd50k_mapping (dict): Mapping from BEATs MID to FSD50K label.
    """

    def __init__(self, scene_dir: str, event_mapping_path: str, fsd50k_meta_dir: str):
        """
        Initialize the ESAS Metadata Generator.

        Args:
            scene_dir (str): Path to CochlScene dataset directory.
            event_mapping_path (str): Path to event_scene_grouping.json file.
            fsd50k_meta_dir (str): Path to FSD50K metadata directory.
        """
        self.scene_dir = Path(scene_dir)
        self.event_mapping = self._load_event_mapping(event_mapping_path)
        self.fsd50k_meta_dir = Path(fsd50k_meta_dir)
        self.scene_classes = list(self.event_mapping.keys())

        # Probability threshold for event presence
        self.probability_threshold = 0.1

        # Load FSD50K labels for event validation
        self.fsd50k_labels = self._load_fsd50k_labels()
        self.fsd50k_vocabulary = self._load_fsd50k_vocabulary()
        self.beats_to_fsd50k_mapping = self._create_beats_to_fsd50k_mapping()

        print(f"FSD50K available events: {len(self.fsd50k_labels)}")
        print(f"BEATs to FSD50K mapping created: {len(self.beats_to_fsd50k_mapping)} mappings")
        print(f"Probability threshold for event presence: >= {self.probability_threshold}")

    def _load_event_mapping(self, mapping_path: str) -> dict:
        """
        Load event-scene mapping from JSON file.

        Args:
            mapping_path (str): Path to the event mapping JSON file.

        Returns:
            dict: Event mapping dictionary with scene-event groupings.
        """
        with open(mapping_path, 'r') as f:
            data = json.load(f)

        # Extract event_groupings from the structure
        return data.get('event_groupings', data)

    def _load_fsd50k_labels(self) -> list:
        """
        Load all available labels from FSD50K metadata (dev.csv and eval.csv).

        Returns:
            list: List of unique sound event labels from FSD50K.
        """
        all_labels = set()
        meta_files = ['dev.csv', 'eval.csv']

        for meta_file in meta_files:
            meta_path = self.fsd50k_meta_dir / meta_file
            if meta_path.exists():
                df = pd.read_csv(meta_path)
                for labels in df['labels']:
                    if isinstance(labels, str):
                        # Safe parsing of label strings (remove brackets and quotes, split by comma)
                        labels_clean = labels.strip("[]").replace("'", "").replace('"', '')
                        label_items = [item.strip() for item in labels_clean.split(",") if item.strip()]
                        all_labels.update(label_items)
            else:
                print(f"Warning: FSD50K metadata file not found: {meta_path}")

        return sorted(list(all_labels))

    def _load_fsd50k_vocabulary(self) -> dict:
        """
        Load FSD50K vocabulary for MID to label mapping.

        Returns:
            dict: Dictionary mapping MID to event label.
        """
        vocab_path = self.fsd50k_meta_dir / "vocabulary.csv"
        vocab_dict = {}

        if vocab_path.exists():
            df = pd.read_csv(vocab_path, header=None, names=['index', 'label', 'mid'])
            for _, row in df.iterrows():
                vocab_dict[row['mid']] = row['label']
            print(f"Loaded FSD50K vocabulary: {len(vocab_dict)} entries")
        else:
            print(f"Warning: FSD50K vocabulary not found: {vocab_path}")

        return vocab_dict

    def _load_beats_class_labels(self) -> dict:
        """
        Load BEATs class labels indices from the model directory.

        Returns:
            dict: Dictionary mapping BEATs indices to label and MID.
        """
        beats_labels_path = Path("model/beats/class_labels_indices.csv")
        beats_dict = {}

        if beats_labels_path.exists():
            df = pd.read_csv(beats_labels_path)
            for _, row in df.iterrows():
                beats_dict[row['index']] = {
                    'label': row['display_name'],
                    'mid': row['mid']
                }
            print(f"Loaded BEATs class labels: {len(beats_dict)} entries")
        else:
            print(f"Warning: BEATs class labels not found: {beats_labels_path}")

        return beats_dict

    def _create_beats_to_fsd50k_mapping(self) -> dict:
        """
        Create mapping from BEATs MID to FSD50K event labels.

        For each BEATs class, it first checks if the MID exists in the FSD50K vocabulary.
        If not, it attempts a fallback matching based on label name similarity.
        If still no match, it uses the original BEATs label.

        Returns:
            dict: Dictionary mapping BEATs MID to FSD50K label.
        """
        beats_labels = self._load_beats_class_labels()
        mapping = {}

        for idx, beats_info in beats_labels.items():
            beats_mid = beats_info['mid']
            beats_label = beats_info['label']

            # Try to find matching FSD50K label by MID
            if beats_mid in self.fsd50k_vocabulary:
                mapping[beats_mid] = self.fsd50k_vocabulary[beats_mid]
            else:
                # Fallback: try to find by label name similarity
                for fsd50k_label in self.fsd50k_labels:
                    if beats_label.lower() in fsd50k_label.lower() or fsd50k_label.lower() in beats_label.lower():
                        mapping[beats_mid] = fsd50k_label
                        break
                else:
                    # If no match found, use original BEATs label
                    mapping[beats_mid] = beats_label

        print(f"Created BEATs to FSD50K mapping: {len(mapping)} mappings")
        return mapping

    def _load_cochlscene_metadata(self) -> list:
        """
        Load CochlScene metadata from BEATs event tags CSV file.

        The CSV file (CochlScene_event_tags_with_BEATs.csv) contains probabilities for each BEATs MID
        per CochlScene recording. This method reads the file, extracts scene label and split from
        the filename, and maps BEATs MIDs to FSD50K labels.

        Returns:
            list: List of dictionaries, each containing clip metadata including all events with
                  probabilities and high-probability events (prob >= threshold).
        """
        cochlscene_metadata = []

        # Load the BEATs event tags CSV file
        beats_csv_path = self.scene_dir / "CochlScene_event_tags_with_BEATs.csv"

        if not beats_csv_path.exists():
            print(f"Error: CochlScene event tags file not found: {beats_csv_path}")
            return cochlscene_metadata

        try:
            df = pd.read_csv(beats_csv_path)
            print(f"Loaded CochlScene event tags: {len(df)} recordings")

            # Get event columns (all columns except filename)
            event_columns = [col for col in df.columns if col != 'filename']
            print(f"Found {len(event_columns)} event columns")

            # Process each recording
            for _, row in df.iterrows():
                filename = row['filename']

                # Extract scene label from filename
                scene_label = self._extract_scene_label(filename)

                # Extract split from filename
                split = self._extract_split(filename)

                # Get all events with their probabilities
                all_events_with_probs = {}
                for event_mid in event_columns:
                    probability = row[event_mid]
                    if pd.notna(probability) and isinstance(probability, (int, float)):
                        # Map BEATs MID to FSD50K label
                        fsd50k_label = self.beats_to_fsd50k_mapping.get(event_mid, event_mid)
                        all_events_with_probs[fsd50k_label] = probability

                # Get high probability events (>= threshold)
                high_prob_events = []
                for event, prob in all_events_with_probs.items():
                    if prob >= self.probability_threshold:
                        high_prob_events.append(event)

                clip_metadata = {
                    'clip_id': filename,
                    'original_scene': filename,
                    'scene_label': scene_label,
                    'all_events_with_probs': all_events_with_probs,  # All events with probabilities
                    'high_probability_events': high_prob_events,  # Events with prob >= threshold
                    'split': split
                }
                cochlscene_metadata.append(clip_metadata)

            print(f"Processed {len(cochlscene_metadata)} recordings")
            print(f"Applied probability threshold: >= {self.probability_threshold} for event presence")

        except Exception as e:
            print(f"Error loading CochlScene event tags: {e}")
            import traceback
            traceback.print_exc()

        return cochlscene_metadata

    def _extract_scene_label(self, filename: str) -> str:
        """
        Extract scene label from filename.

        Expected filename format: "split/scene_name_rest.wav" (e.g., "train/Airport_01.wav").

        Args:
            filename (str): Audio filename.

        Returns:
            str: Extracted scene label.
        """
        parts = filename.split('/')
        if len(parts) >= 2:
            scene_part = parts[1]
            # Remove duplicate scene name if present
            if '_' in scene_part:
                scene_label = scene_part.split('_')[0]
            else:
                scene_label = scene_part
            return scene_label
        return 'unknown'

    def _extract_split(self, filename: str) -> str:
        """
        Extract dataset split from filename.

        The split is assumed to be the first part of the path (e.g., "Val", "test", "train").

        Args:
            filename (str): Audio filename.

        Returns:
            str: Extracted split ('train', 'val', 'test') normalized to lowercase.
        """
        parts = filename.split('/')
        if len(parts) >= 1:
            split_part = parts[0].lower()  # "Val" -> "val"
            if split_part == 'val':
                return 'val'
            elif split_part == 'test':
                return 'test'
            else:
                return 'train'
        return 'unknown'

    def filter_real_unknown_recordings(self, cochlscene_metadata: list) -> tuple:
        """
        Filter CochlScene recordings to identify those containing scene-specific unknown events.

        This function checks each recording's high-probability events against the list of
        unknown events defined for its scene in the event mapping. Recordings that contain
        any unknown event with probability >= threshold are marked as "real-unknown".

        Args:
            cochlscene_metadata (list): List of dictionaries containing CochlScene metadata.

        Returns:
            tuple: (filtered_metadata, real_unknown_metadata) where:
                - filtered_metadata: recordings containing only known events or no events.
                - real_unknown_metadata: recordings containing scene-specific unknown events.
        """
        filtered_metadata = []
        real_unknown_metadata = []

        unknown_detection_count = 0
        scene_unknown_stats = {}
        split_unknown_stats = {'train': 0, 'val': 0, 'test': 0}

        print(f"\nStarting real-unknown detection:")
        print(f"  - Probability threshold: >= {self.probability_threshold}")
        print(f"  - Checking scene-specific unknown events from event_scene_grouping.json")

        for clip_metadata in cochlscene_metadata:
            scene_label = clip_metadata.get('scene_label')
            split = clip_metadata.get('split', 'unknown')
            high_prob_events = clip_metadata.get('high_probability_events', [])
            all_events_with_probs = clip_metadata.get('all_events_with_probs', {})

            if scene_label not in self.event_mapping:
                # If scene not in mapping, include in filtered set
                filtered_metadata.append(clip_metadata)
                continue

            # Get unknown events specific to this scene
            scene_unknown_events = self.event_mapping[scene_label].get('unknown_events', [])

            # Check if recording contains any scene-specific unknown events with probability >= threshold
            unknown_events_found = []
            unknown_events_detailed = []

            for unknown_event in scene_unknown_events:
                if unknown_event in all_events_with_probs:
                    probability = all_events_with_probs[unknown_event]
                    if probability >= self.probability_threshold:  # Event is present
                        unknown_events_found.append(unknown_event)
                        unknown_events_detailed.append({
                            'event': unknown_event,
                            'probability': probability
                        })

            contains_unknown = len(unknown_events_found) > 0

            if contains_unknown:
                unknown_detection_count += 1

                # Update statistics
                if scene_label not in scene_unknown_stats:
                    scene_unknown_stats[scene_label] = 0
                scene_unknown_stats[scene_label] += 1

                if split in split_unknown_stats:
                    split_unknown_stats[split] += 1

                # Separate known events (high probability events that are not unknown for this scene)
                known_events_found = []
                for event in high_prob_events:
                    if event not in scene_unknown_events:
                        known_events_found.append(event)

                # Mark as real unknown
                enhanced_metadata = clip_metadata.copy()
                enhanced_metadata['event_labels'] = known_events_found  # Only known events
                enhanced_metadata['mix_type'] = 'real-unknown'
                enhanced_metadata['unknown_events_found'] = unknown_events_found
                enhanced_metadata['unknown_events_detailed'] = unknown_events_detailed
                enhanced_metadata['event_snr'] = []  # Real recordings have no synthetic SNR
                enhanced_metadata['event_timestamps'] = []  # Real recordings have no synthetic timestamps
                enhanced_metadata['duration'] = 10.0  # Standard duration
                enhanced_metadata['sampling_rate'] = 44100  # Standard sampling rate
                real_unknown_metadata.append(enhanced_metadata)
            else:
                # Include in main dataset with appropriate mix type
                enhanced_metadata = clip_metadata.copy()
                enhanced_metadata['event_labels'] = high_prob_events  # All high probability events

                if len(high_prob_events) == 0:
                    enhanced_metadata['mix_type'] = 'background-only'
                else:
                    enhanced_metadata['mix_type'] = 'known-event'
                enhanced_metadata['event_snr'] = []
                enhanced_metadata['event_timestamps'] = []
                enhanced_metadata['duration'] = 10.0
                enhanced_metadata['sampling_rate'] = 44100
                filtered_metadata.append(enhanced_metadata)

        print(f"\nReal unknown filtering completed:")
        print(f"  - Total recordings processed: {len(cochlscene_metadata)}")
        print(f"  - Filtered recordings (no unknown events): {len(filtered_metadata)}")
        print(f"  - Real unknown recordings (contain unknown events): {len(real_unknown_metadata)}")
        print(f"  - Unknown event detections: {unknown_detection_count}")

        # Print split-wise statistics
        print("\nSplit-wise real-unknown distribution:")
        for split, count in split_unknown_stats.items():
            total_in_split = len([clip for clip in cochlscene_metadata if clip.get('split') == split])
            percentage = (count / total_in_split * 100) if total_in_split > 0 else 0
            print(f"  - {split}: {count} recordings ({percentage:.1f}% of {total_in_split} total)")

        # Print scene-wise statistics
        if scene_unknown_stats:
            print("\nScene-wise real-unknown distribution:")
            for scene, count in sorted(scene_unknown_stats.items(), key=lambda x: x[1], reverse=True):
                total_in_scene = len([clip for clip in cochlscene_metadata if clip.get('scene_label') == scene])
                percentage = (count / total_in_scene * 100) if total_in_scene > 0 else 0
                unknown_events = self.event_mapping[scene].get('unknown_events', [])
                print(f"  - {scene}: {count} recordings ({percentage:.1f}% of {total_in_scene} total)")
                print(f"    Unknown events for this scene: {unknown_events}")

        # Debug: print some examples of found unknown events
        if real_unknown_metadata:
            print("\nExamples of real-unknown recordings:")
            for i, clip in enumerate(real_unknown_metadata[:5]):
                scene = clip['scene_label']
                split = clip.get('split', 'unknown')
                unknown_events_found = clip.get('unknown_events_found', [])
                detailed_unknown = clip.get('unknown_events_detailed', [])
                known_events = clip.get('event_labels', [])
                print(f"  {i + 1}. {clip['clip_id']}")
                print(f"     Scene: {scene}, Split: {split}")
                print(f"     Known events found: {known_events}")
                print(f"     Unknown events found (prob >= {self.probability_threshold}): {unknown_events_found}")
                for detail in detailed_unknown:
                    print(f"        {detail['event']}: {detail['probability']:.6f}")

        return filtered_metadata, real_unknown_metadata

    def generate_real_unknown_metadata(self, output_dir: str = "metadata") -> list:
        """
        Generate metadata for real-unknown recordings and save to CSV.

        This method loads CochlScene metadata, filters for real-unknown recordings,
        and writes the results to 'real_unknown_metadata.csv' in the specified output directory.
        It also saves statistics to 'real_unknown_statistics.json' in the 'docs' folder.

        Args:
            output_dir (str): Output directory for real-unknown metadata.

        Returns:
            list: List of real-unknown metadata dictionaries.
        """
        print("\n" + "=" * 60)
        print("Generating real-unknown metadata...")
        print("=" * 60)

        # Load CochlScene metadata
        cochlscene_metadata = self._load_cochlscene_metadata()

        if not cochlscene_metadata:
            print("No CochlScene metadata found. Skipping real-unknown generation.")
            return []

        # Filter real unknown recordings
        filtered_metadata, real_unknown_metadata = self.filter_real_unknown_recordings(cochlscene_metadata)

        # Save real-unknown metadata
        if real_unknown_metadata:
            # Create clean metadata for CSV output
            clean_real_unknown = []
            for clip in real_unknown_metadata:
                clean_clip = {
                    'clip_id': clip['clip_id'],
                    'original_scene': clip['original_scene'],
                    'scene_label': clip['scene_label'],
                    'event_labels': clip.get('event_labels', []),
                    'split': clip.get('split', 'unknown'),
                    'mix_type': clip.get('mix_type', 'real-unknown'),
                    'unknown_events_found': clip.get('unknown_events_found', []),
                    'event_snr': clip.get('event_snr', []),
                    'event_timestamps': clip.get('event_timestamps', []),
                    'duration': clip.get('duration', 10.0),
                    'sampling_rate': clip.get('sampling_rate', 44100)
                }
                clean_real_unknown.append(clean_clip)

            real_unknown_df = pd.DataFrame(clean_real_unknown)
            real_unknown_path = Path(output_dir) / "real_unknown_metadata.csv"
            real_unknown_df.to_csv(real_unknown_path, index=False)
            print(f"Real-unknown metadata saved to: {real_unknown_path}")

            # Save statistics
            real_unknown_stats = self._analyze_real_unknown_stats(real_unknown_metadata)
            stats_path = Path("docs") / "real_unknown_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(real_unknown_stats, f, indent=2)
            print(f"Real-unknown statistics saved to: {stats_path}")

        else:
            print("No real-unknown recordings found.")
            # Create empty real-unknown metadata file for consistency
            real_unknown_df = pd.DataFrame(columns=[
                'clip_id', 'original_scene', 'scene_label', 'event_labels',
                'split', 'mix_type', 'unknown_events_found',
                'event_snr', 'event_timestamps', 'duration', 'sampling_rate'
            ])
            real_unknown_path = Path(output_dir) / "real_unknown_metadata.csv"
            real_unknown_df.to_csv(real_unknown_path, index=False)
            print(f"Empty real-unknown metadata saved to: {real_unknown_path}")

        return real_unknown_metadata

    def _analyze_real_unknown_stats(self, real_unknown_metadata: list) -> dict:
        """
        Analyze statistics for real-unknown recordings.

        Computes distribution by scene and split, and collects probability statistics
        of the unknown events found.

        Args:
            real_unknown_metadata (list): List of real-unknown metadata dictionaries.

        Returns:
            dict: Statistics for real-unknown recordings.
        """
        stats = {
            'total_recordings': len(real_unknown_metadata),
            'probability_threshold': self.probability_threshold,
            'scene_distribution': {},
            'split_distribution': {},
            'unknown_events_distribution': {},
            'probability_statistics': {}
        }

        all_probabilities = []

        for clip in real_unknown_metadata:
            scene = clip['scene_label']
            split = clip.get('split', 'unknown')

            # Update scene distribution
            if scene not in stats['scene_distribution']:
                stats['scene_distribution'][scene] = 0
            stats['scene_distribution'][scene] += 1

            # Update split distribution
            if split not in stats['split_distribution']:
                stats['split_distribution'][split] = 0
            stats['split_distribution'][split] += 1

            # Update unknown events distribution and collect probabilities
            for event_detail in clip.get('unknown_events_detailed', []):
                event = event_detail['event']
                probability = event_detail['probability']

                if event not in stats['unknown_events_distribution']:
                    stats['unknown_events_distribution'][event] = 0
                stats['unknown_events_distribution'][event] += 1

                all_probabilities.append(probability)

        # Probability statistics
        if all_probabilities:
            stats['probability_statistics'] = {
                'mean': float(np.mean(all_probabilities)),
                'std': float(np.std(all_probabilities)),
                'min': float(np.min(all_probabilities)),
                'max': float(np.max(all_probabilities)),
                'median': float(np.median(all_probabilities)),
                'threshold': self.probability_threshold,
                'description': f'Events with probability >= {self.probability_threshold} are considered present'
            }

        return stats

    def validate_event_mapping(self) -> tuple:
        """
        Validate that all events in mapping exist in FSD50K.

        Returns:
            tuple: (valid_events, invalid_events) – lists of valid and invalid event names.
        """
        valid_events = []
        invalid_events = []

        for scene_label, event_groups in self.event_mapping.items():
            # Validate known events
            for event in event_groups.get('known_events', []):
                if event in self.fsd50k_labels:
                    valid_events.append(event)
                else:
                    invalid_events.append(event)

            # Validate unknown events
            for event in event_groups.get('unknown_events', []):
                if event in self.fsd50k_labels:
                    valid_events.append(event)
                else:
                    invalid_events.append(event)

        print(f"Event mapping validation:")
        print(f"Valid events: {len(set(valid_events))}")
        print(f"Invalid events: {len(set(invalid_events))}")

        if invalid_events:
            print(f"Invalid event examples: {list(set(invalid_events))[:5]}")

        return list(set(valid_events)), list(set(invalid_events))

    def generate_event_splits(self) -> dict:
        """
        Generate known/unknown event splits based on the mapping.

        Returns:
            dict: Dictionary containing:
                - 'known_events': sorted list of known event labels.
                - 'unknown_events': sorted list of unknown event labels.
                - 'statistics': summary counts and ratios.
        """
        # Get all valid events
        valid_events, invalid_events = self.validate_event_mapping()

        if len(valid_events) == 0:
            raise ValueError("No valid events found. Please check event mapping and FSD50K data.")

        # Collect known and unknown events
        known_events = set()
        unknown_events = set()

        for scene_label, event_groups in self.event_mapping.items():
            for event in event_groups.get('known_events', []):
                if event in self.fsd50k_labels:
                    known_events.add(event)

            for event in event_groups.get('unknown_events', []):
                if event in self.fsd50k_labels:
                    unknown_events.add(event)

        known_events = sorted(list(known_events))
        unknown_events = sorted(list(unknown_events))

        event_splits = {
            'known_events': known_events,
            'unknown_events': unknown_events,
            'statistics': {
                'total_valid_events': len(valid_events),
                'known_count': len(known_events),
                'unknown_count': len(unknown_events),
                'known_ratio': len(known_events) / len(valid_events) if valid_events else 0.0,
                'invalid_events_count': len(invalid_events)
            }
        }

        return event_splits

    def create_metadata_schema(self) -> dict:
        """
        Create metadata schema definition for ESAS dataset.

        Returns:
            dict: Metadata schema specification with field names, types, and descriptions.
        """
        schema = {
            'clip_id': {
                'type': 'string',
                'description': 'Unique identifier for the audio clip',
                'format': '{scene_label}_{split}_{index:06d}'
            },
            'original_scene': {
                'type': 'string',
                'description': 'Filename of the original CochlScene audio'
            },
            'scene_label': {
                'type': 'string',
                'description': 'Acoustic scene class label',
                'values': self.scene_classes
            },
            'event_labels': {
                'type': 'list[string]',
                'description': 'List of sound event labels added to the scene'
            },
            'event_snr': {
                'type': 'list[float]',
                'description': 'Signal-to-noise ratio for each event in dB',
                'range': '[-5, 5]'
            },
            'event_timestamps': {
                'type': 'list[list[float]]',
                'description': 'Start and end times for each event in seconds'
            },
            'split': {
                'type': 'string',
                'description': 'Dataset split',
                'values': ['train', 'val', 'test', 'real-unknown']
            },
            'mix_type': {
                'type': 'string',
                'description': 'Type of audio mixture',
                'values': ['background-only', 'known-event', 'unknown-event', 'real-unknown']
            },
            'duration': {
                'type': 'float',
                'description': 'Audio clip duration in seconds',
                'value': 10.0
            },
            'sampling_rate': {
                'type': 'integer',
                'description': 'Audio sampling rate in Hz',
                'value': 44100
            }
        }
        return schema

    def generate_metadata_template(self, output_path: str) -> None:
        """
        Generate CSV template for metadata.

        Creates an empty CSV file with the expected columns for ESAS metadata.

        Args:
            output_path (str): Output path for the metadata template CSV file.
        """
        # Create directory structure
        Path("metadata").mkdir(exist_ok=True)
        Path("docs").mkdir(exist_ok=True)

        # Create template DataFrame
        template_data = {
            'clip_id': [],
            'original_scene': [],
            'scene_label': [],
            'event_labels': [],
            'event_snr': [],
            'event_timestamps': [],
            'split': [],
            'mix_type': [],
            'duration': [],
            'sampling_rate': []
        }

        df = pd.DataFrame(template_data)
        df.to_csv(output_path, index=False)
        print(f"Metadata template saved to: {output_path}")

    def analyze_scene_event_coverage(self) -> dict:
        """
        Analyze event coverage statistics per scene.

        For each scene, counts the number of known and unknown events defined in the mapping
        that also exist in FSD50K.

        Returns:
            dict: Coverage statistics for each scene, including event lists.
        """
        coverage_stats = {}

        for scene_label in self.scene_classes:
            known_events = [e for e in self.event_mapping[scene_label].get('known_events', []) if
                            e in self.fsd50k_labels]
            unknown_events = [e for e in self.event_mapping[scene_label].get('unknown_events', []) if
                              e in self.fsd50k_labels]
            all_events = known_events + unknown_events

            coverage_stats[scene_label] = {
                'known_events_count': len(known_events),
                'unknown_events_count': len(unknown_events),
                'total_events_count': len(all_events),
                'known_events': known_events,
                'unknown_events': unknown_events
            }

        return coverage_stats

    def generate_all_metadata(self) -> dict:
        """
        Generate complete metadata package for ESAS dataset.

        This method orchestrates the generation of:
            - metadata_template.csv
            - event_splits.json
            - metadata_schema.json
            - scene_event_coverage.json
            - real_unknown_metadata.csv
            - real_unknown_statistics.json

        Returns:
            dict: Event splits with statistics.
        """
        print("Generating ESAS Dataset Metadata")
        print("=" * 50)

        # 1. Generate metadata template
        print("1. Generating metadata template...")
        self.generate_metadata_template("data/metadata/metadata_template.csv")

        # 2. Generate event splits
        print("2. Generating event splits...")
        event_splits = self.generate_event_splits()

        with open('docs/event_splits.json', 'w') as f:
            json.dump(event_splits, f, indent=2)

        # 3. Generate metadata schema
        print("3. Creating metadata schema...")
        schema = self.create_metadata_schema()
        with open('docs/metadata_schema.json', 'w') as f:
            json.dump(schema, f, indent=2)

        # 4. Analyze scene-event coverage
        print("4. Analyzing scene-event coverage...")
        coverage_stats = self.analyze_scene_event_coverage()
        with open('docs/scene_event_coverage.json', 'w') as f:
            json.dump(coverage_stats, f, indent=2)

        # 5. Generate real-unknown metadata
        print("5. Generating real-unknown metadata...")
        real_unknown_metadata = self.generate_real_unknown_metadata()

        # Output results
        stats = event_splits['statistics']
        print(f"\nMetadata generation completed!")
        print(f"Scene classes: {len(self.scene_classes)}")
        print(f"Valid events: {stats['total_valid_events']}")
        print(f"Known events: {stats['known_count']}")
        print(f"Unknown events: {stats['unknown_count']}")
        print(f"Known ratio: {stats['known_ratio']:.1%}")
        print(f"Invalid events: {stats['invalid_events_count']}")
        print(f"Real-unknown recordings: {len(real_unknown_metadata)}")

        return event_splits


def main():
    """
    Main execution function for ESAS metadata generation.

    This function checks for required input files, initializes the metadata generator,
    and runs the full metadata generation pipeline.
    """
    print("ESAS Dataset Metadata Generator with Real-Unknown Detection")
    print("=" * 60)
    print("Detection Logic:")
    print("  1. Load CochlScene event tags with BEATs probabilities")
    print("  2. Map BEATs labels to FSD50K labels")
    print("  3. Apply probability threshold: >= 0.1 for event presence")
    print("  4. Check for scene-specific unknown events from event_scene_grouping.json")
    print("  5. Mark recordings with unknown events as 'real-unknown'")
    print("=" * 60)

    # Check required files
    required_files = [
        "docs/event_scene_grouping.json"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease run previous scripts to create these files first.")
        return

    try:
        # Initialize metadata generator
        generator = ESASMetadataGenerator(
            scene_dir=".../CochlScene",
            event_mapping_path="docs/event_scene_grouping.json",
            fsd50k_meta_dir=".../FSD50K/FSD50K.ground_truth"
        )

        # Generate complete metadata package
        generator.generate_all_metadata()

        print(f"\nGenerated files:")
        print("  - data/metadata/metadata_template.csv")
        print("  - data/metadata/real_unknown_metadata.csv")
        print("  - docs/event_splits.json")
        print("  - docs/metadata_schema.json")
        print("  - docs/scene_event_coverage.json")
        print("  - docs/real_unknown_statistics.json")

    except Exception as e:
        print(f"Error during metadata generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()