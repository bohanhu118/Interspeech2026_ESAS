# Towards Event-Robust Acoustic Scene Classification - Train Model with YAML files
Author: **Yiqiang Cai** (yiqiang.cai21@student.xjtlu.edu.cn), **Bohan Hu** (bohan.hu24@student.xjtlu.edu.cn), *Xi'an Jiaotong-Liverpool University*

##  Work Description
* The Event-Shifted Acoustic Scene (ESAS) dataset, a novel benchmark for evaluating the robustness of Acoustic Scene Classification (ASC) systems against unknown sound events. Existing ASC datasets typically contain recordings of clean and consistent audio, while real-world environments often include diverse and unexpected sound events.
* To bridge this gap, ESAS simulates real-world acoustic variability by injecting foreground sound events into background scenes with the assistance of large language models. 
* In this work, we present the construction methodology, dataset statistics, and evaluation protocols. Furthermore, a comprehensive evaluation of state-of-the-art ASC systems is conducted using the ESAS benchmark. Experimental results reveal that existing ASC models suffer significant performance degradation when facing the event-shift challenge.
* The introduction of the ESAS dataset aims to drive future research toward event-robust ASC.

## System Description
This repository provides an easy way to train your models on the ESAS dataset.

1. All configurations of model, dataset and training can be done via a simple YAML file.
2. Entire system is implemented using [PyTorch Lightning](https://lightning.ai/).
3. Logging is implemented using [TensorBoard](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.TensorBoardLogger.html#tensorboardlogger). ([Wandb API](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html) is also supported.)
4. Various task-related techniques have been included.
   * 4 Spectrogram Extractor: Cnn3Mel, CpMel, BEATsMel, PaSSTMel.
   * 6 High-performing Backbones: BEATs, PASST, TF-SepNet, BC-ResNet, CP-Mobile, GRU-CNN.
   * 5 Plug-and-played Data Augmentation Techniques: MixUp, MixUpMultiLabels, FreqMixStyle, SpecAugmentation, Device Impulse Response Augmentation.

## Getting Started

1. Clone this repository.
2. Create and activate a [conda](https://docs.anaconda.com/free/miniconda/index.html) environment:

```
conda create -n ESAS
conda activate ESAS
```

3. Install [PyTorch](https://pytorch.org/get-started/previous-versions/) version that suits your system. For example:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or for cuda >= 12.1
pip install torch torchvision torchaudio
```

4. Install requirements:

```
pip install -r requirements.txt
```

5. Download and extract the Event-Shifted Acoustic Scene (ESAS) dataset according to your needs. The directory should be placed in the **parent path** of code directory.

You should end up with a directory that contains, among other files, the following:
* ../esas_data/: A directory containing audio files in *wav* format.

6. Several default configuration yaml files are provided in config/. The training procedure can be started by running the following command:
```
python -m main fit --config config/cpmobile/cpmobile_train.yaml
```

7. Test model:
```
python -m main test --config config/cpmobile/cpmobile_test.yaml --ckpt_path path/to/ckpt
```

8. View results:
```
tensorboard --logdir log/cpmobile_train  # Check training results
tensorboard --logdir log/cpmobile_test  # Check testing results
```
Then results will be available at [localhost port 6006](http://127.0.0.1:6006/).

## Dataset Generation
In addition to downloading the pre-generated ESAS dataset, you can also build it from scratch using the provided pipeline. The generation process consists of three main stages:

1. **Audio Event Tagging** – Use the BEATs model to predict event probabilities for each CochlScene recording.
2. **Metadata Generation** – Create the ESAS metadata, identify “real-unknown” recordings, and define known/unknown event splits.
3. **Audio Mixing** – Synthesise the final dataset by mixing background scenes with foreground events according to the ESAS protocol.

All scripts are located in the root directory of the repository.


**Requirements:**
- CochlScene dataset placed in a known directory (e.g., `../CochlScene`).
- BEATs model checkpoint (e.g., `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`) downloaded and placed in an accessible location.
- FSD50K metadata (for label mapping) available in `.../FSD50K/FSD50K.ground_truth`.


***Script:*** `audio_tagging.py`

Before running, open the script and adjust the following variables to match your local paths:
```python
audio_dir = ".../CochlScene"   # Path to CochlScene root
ckpt_event_model = ".../BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"   # BEATs checkpoint
fsd50k_meta_dir = ".../FSD50K/FSD50K.ground_truth"   # FSD50K metadata directory
```

The script processes all audio files in `audio_dir` and saves a CSV file named `CochlScene_event_tags_with_BEATs.csv` in the same directory. 
This file contains one row per CochlScene recording, with columns for each BEATs class MID storing the predicted probability.

***Script:*** `generate_metadata.py`

Edit the paths in the main() function if necessary (the defaults are shown below):

```python
generator = ESASMetadataGenerator(
    scene_dir=".../CochlScene",
    event_mapping_path="docs/event_scene_grouping.json",
    fsd50k_meta_dir=".../FSD50K/FSD50K.ground_truth"
)
```

Output files are created in the following locations:

`data/metadata/metadata_template.csv` – empty template with the correct columns.
`data/metadata/real_unknown_metadata.csv` – list of real‑unknown recordings.
`docs/event_splits.json` – known/unknown event lists and statistics.
`docs/metadata_schema.json` – field descriptions for the metadata.
`docs/scene_event_coverage.json` – per‑scene event counts.
`docs/real_unknown_statistics.json` – distribution of real‑unknown recordings.

***Script***: `mix_audio.py`

The mixer supports three mix types:
**background‑only** – only the original scene audio.
**known‑event** – scene mixed with known events (from the training split of FSD50K).
**syth‑unknown** – scene mixed with unknown events (from the evaluation split of FSD50K).
Events are placed with allowed overlap, and each event’s audio may be time‑stretched and pitch‑shifted for variation. The mixer also tracks file usage to ensure balanced sampling and prevent excessive reuse of the same event file.

To generate the dataset for a specific split, run:
```
python esas_mixer.py --split {train,val,test} --clips_per_scene N [--max_reuse_per_file M]
```

The generated audio files are saved under ``data/esas_data/{split}/`` with filenames like ``{scene_label}_{split}_{...}.wav``. Metadata for the split is written to ``data/metadata/{split}.csv``.

After all three splits have been generated, the folder structure should look like:
```
../esas_data/
├── train/           # background-only + known-event clips
├── val/             # background-only + known-event clips
├── test/            # background-only + known-event + syth-unknown clips
└── metadata/        # per-split CSV files
```

## Customize Your System

Deploy your model in `model/backbones/` and inherit the **_BaseBackbone**:
```
class YourModel(_BaseBackbone):
...
```
Implement new spectrogram extractor in `util/spec_extractor/` and inherit the **_SpecExtractor**:
```
class NewExtractor(_SpecExtractor):
...
```
Declare new data augmentation method in `util/data_augmentation/` and inherit the **_DataAugmentation**:  
```
class NewAugmentation(_DataAugmentation):
...
```

More instructions can be found on [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)
