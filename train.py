#!/usr/bin/env python
import os
import shutil
import os.path as osp
from collections import OrderedDict
import pickle
import yaml
import time
import torch
from torch import nn
from transformers import AdamW, AlbertConfig, AlbertModel
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import wandb

from model import MultiTaskModel
from dataloader import build_dataloader
from utils import length_to_mask
from datasets import load_from_disk

config_path = "Configs/config.yml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

with open(config['dataset_params']['token_maps'], 'rb') as handle:
    token_maps = pickle.load(handle)

criterion = nn.CrossEntropyLoss()
num_steps = config['num_steps']
log_interval = config['log_interval']
save_interval = config['save_interval']


def train():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    curr_steps = 0

    dataset = load_from_disk(config["data_folder"])

    log_dir = config['log_dir']
    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

    train_loader = build_dataloader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=16,
        dataset_config=config['dataset_params'],
    )

    albert_base_configuration = AlbertConfig(**config['model_params'])

    bert = AlbertModel(albert_base_configuration)
    bert = MultiTaskModel(
        bert,
        num_vocab=1 + max([m['token'] for m in token_maps.values()]),
        num_tokens=config['model_params']['vocab_size'],
        hidden_size=config['model_params']['hidden_size']
    )

    load = True
    try:
        files = os.listdir(log_dir)
        ckpts = []
        for f in os.listdir(log_dir):
            if f.startswith("step_"):
                ckpts.append(f)

        iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
        iters = sorted(iters)[-1]
    except:
        iters = 0
        load = False

    optimizer = AdamW(bert.parameters(), lr=1e-4)

    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        split_batches=False,
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        wandb.init(
            project="PL-BERT_ar",
            name="run_" + time.strftime("%Y%m%d-%H%M%S"),
            config=config
        )

        accelerator.print("Dataset size =", len(dataset))
        accelerator.print("Batch size =", config["batch_size"])
        accelerator.print("Num processes =", accelerator.num_processes)

    if load:
        checkpoint = torch.load(log_dir + "/step_" + str(iters) + ".t7", map_location='cpu')
        state_dict = checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v

        bert.load_state_dict(new_state_dict, strict=False)
        if accelerator.is_main_process:
            accelerator.print('Checkpoint loaded.')
        optimizer.load_state_dict(checkpoint['optimizer'])

    bert, optimizer, train_loader = accelerator.prepare(
        bert, optimizer, train_loader
    )

    accelerator.print('Start training...')

    running_loss = 0.0

    epoch = 0
    while curr_steps < num_steps:
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        for _, batch in enumerate(train_loader):
            curr_steps += 1

            words, labels, phonemes, input_lengths, masked_positions = batch
            device = phonemes.device
            input_lengths_tensor = torch.tensor(input_lengths, device=device)
            text_mask = length_to_mask(input_lengths_tensor).to(phonemes.device)

            tokens_pred, words_pred = bert(phonemes, attention_mask=(~text_mask).int())
            B, T, Vv = words_pred.shape

            words_pred_flat = words_pred.view(B * T, Vv)
            words_flat = words.view(B * T)
            valid_flat = (~text_mask).view(B * T)

            if valid_flat.any():
                loss_vocab = criterion(words_pred_flat[valid_flat], words_flat[valid_flat])
            else:
                loss_vocab = torch.tensor(0.0, device=device)

            Bt, Tt, Vt = tokens_pred.shape
            mask_tensor = torch.zeros((B, T), dtype=torch.bool, device=device)
            for i, mp in enumerate(masked_positions):
                mask_tensor[i, :mp.size(0)] = mp

            tokens_pred_flat = tokens_pred.view(Bt * Tt, Vt)
            labels_flat = labels.view(Bt * Tt)
            mask_flat = mask_tensor.view(-1)

            if mask_flat.any():
                loss_token = criterion(tokens_pred_flat[mask_flat], labels_flat[mask_flat])
            else:
                loss_token = torch.tensor(0.0, device=device)

            loss = loss_vocab + loss_token

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()

            iters = iters + 1
            if (iters + 1) % log_interval == 0:
                avg_loss = running_loss / float(log_interval)
                accelerator.print(
                    'Step [%d/%d], Loss: %.5f, Vocab Loss: %.5f, Token Loss: %.5f'
                    % (
                        iters + 1,
                        num_steps,
                        avg_loss,
                        float(loss_vocab.detach().item()) if isinstance(loss_vocab, torch.Tensor) else float(loss_vocab),
                        float(loss_token.detach().item()) if isinstance(loss_token, torch.Tensor) else float(loss_token)
                    )
                )

                if accelerator.is_main_process:
                    wandb.log({
                        "step": iters + 1,
                        "Avg Loss": avg_loss,
                        "Vocab Loss": float(loss_vocab.detach().item()) if isinstance(loss_vocab, torch.Tensor) else float(loss_vocab),
                        "Token Loss": float(loss_token.detach().item()) if isinstance(loss_token, torch.Tensor) else float(loss_token),
                    })

                running_loss = 0.0

            if (iters + 1) % save_interval == 0:
                if accelerator.is_main_process:
                    accelerator.print('Saving..')

                state = {
                    'net': bert.state_dict(),
                    'step': iters,
                    'optimizer': optimizer.state_dict(),
                }

                accelerator.save(state, log_dir + '/step_' + str(iters + 1) + '.t7')

                if accelerator.is_main_process:
                    wandb.log({"checkpoint": iters + 1})

                accelerator.wait_for_everyone()

            if curr_steps >= num_steps:
                if accelerator.is_main_process:
                    accelerator.print("Reached max training steps:", curr_steps)
                break

        epoch += 1
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.finish()
    accelerator.print("Training loop exited cleanly. Total steps:", curr_steps)


if __name__ == "__main__":
    train()
