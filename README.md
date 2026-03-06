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
