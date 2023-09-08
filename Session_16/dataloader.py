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
            filtered_train_ds = [k for k in train_ds_raw if len(k["translation"][config["lang_src"]])<=150] # and (len(k["translation"][config["lang_tgt"]])<=len(k["translation"][config["lang_src"]]) + 10)]
            filtered_train_ds = [k for k in filtered_train_ds if len(k["translation"][config["lang_tgt"]])<len(k["translation"][config["lang_src"]]) + 10]
            self.train_dataset = BilingualDataset(
                filtered_train_dstrain_ds_raw,
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
            dataset=self.train_dataset, batch_size=self.config["batch_size"], shuffle=True,collate_fn=collate_fn
        )
    def collate_fn(batch):
        encoder_input_max=max(x["encoder_str_length"] for x in batch)
        decoder_input_max=max(x["decoder_str_length"] for x in batch)
        encoder_inputs=[]
        decoder_inputs=[]
        encoder_mask=[]
        decoder_mask=[]
        label=[]
        src_text=[]
        tgt_text=[]
        for b in batch:
            encoder_inputs.append(b["encoder_input"][:encoder_input_max])
            decoder_inputs.append(b["decoder_input"][:decoder_input_max])
            encoder_mask.append(b["encoder_mask"][0,0,:encoder_input_max]).unsqueeze(0).unsqueeze(0).unsqueeze(0))
            decoder_mask.append(b["decoder_mask"][0,0:decoder_input_max,:decoder_input_max]).unsqueeze(0).unsqueeze(0))
            label.append(b["label"][:decoder_input_max])
            src_text.append(b["src_text"])
            tgt_text.append(b["tgt_text"])
        return {
            "encoder_input":torch.vstack(encoder_inputs),
            "decoder_input":torch.vstack(decoder_inputs),
            "encoder_mask":torch.vstack(encoder_mask),
            "decoder_mask":torch.vstack(decoder_mask),
            "label":torch.vstack(label),
            "src_txt":src_text,
            "tgt_txt":tgt_text

        }

    

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=1, shuffle=False)
