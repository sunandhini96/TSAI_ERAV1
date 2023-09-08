from pathlib import Path
import os

from tokenizers import Tokenizer

from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import BilingualDataset
from utils import get_or_build_tokenizer
from config import get_config

config = get_config()

class OpusDataSetModule(LightningDataModule):
    def __init__(self, config=config):
        super().__init__()

        self.config = config
        self.train_dataset = None
        self.val_dataset = None

        self.tokenizer_src = None
        self.tokenizer_tgt = None

    def prepare_data(self):
        load_dataset(
            "opus_books",
            f"{self.config['lang_src']}-{self.config['lang_tgt']}",
            split="train",
        )

    def setup(self, stage=None):
        if not self.train_dataset and not self.val_dataset:
            ds_raw = load_dataset(
                "opus_books",
                f"{self.config['lang_src']}-{self.config['lang_tgt']}",
                split="train",
            )

            # Build tokenizers
            self.tokenizer_src = get_or_build_tokenizer(
                self.config, ds_raw, self.config["lang_src"]
            )
            self.tokenizer_tgt = get_or_build_tokenizer(
                self.config, ds_raw, self.config["lang_tgt"]
            )

            # keep 90% for training, 10% for validation
            train_ds_size = int(0.9 * len(ds_raw))
            val_ds_size = len(ds_raw) - train_ds_size
            train_ds_raw, val_ds_raw = random_split(
                ds_raw, [train_ds_size, val_ds_size]
            )

            self.train_dataset = BilingualDataset(
                train_ds_raw,
                self.tokenizer_src,
                self.tokenizer_tgt,
                self.config["lang_src"],
                self.config["lang_tgt"],
                self.config["seq_len"],
            )

            self.val_dataset = BilingualDataset(
                val_ds_raw,
                self.tokenizer_src,
                self.tokenizer_tgt,
                self.config["lang_src"],
                self.config["lang_tgt"],
                self.config["seq_len"],
            )

            # Find the max length of each sentence in the source and target sentnece
            max_len_src = 0
            max_len_tgt = 0

            for item in ds_raw:
                src_ids = self.tokenizer_src.encode(
                    item["translation"][self.config["lang_src"]]
                ).ids
                tgt_ids = self.tokenizer_tgt.encode(
                    item["translation"][self.config["lang_tgt"]]
                ).ids
                max_len_src = max(max_len_src, len(src_ids))
                max_len_tgt = max(max_len_tgt, len(tgt_ids))

            print(f"Max length of source sentence: {max_len_src}")
            print(f"Max length of target sentence: {max_len_tgt}")

            print(f"Source Tokenizer Vocab Size : {self.tokenizer_src.get_vocab_size()}")
            print(f"Target Tokenizer Vocab Size : {self.tokenizer_tgt.get_vocab_size()}")
            print("\n")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.config["batch_size"], shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=1, shuffle=False)
