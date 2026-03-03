# main.py
from lightning.pytorch.cli import LightningCLI
import torch
import model.lit_model
import data.data_module_cochlscene
import util
import model.backbone

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    cli = LightningCLI()

if __name__ == '__main__':
    main()
