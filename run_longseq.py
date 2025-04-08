# Written by Yukang Chen
# Some code based on https://github.com/epfml/landmark-attention
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
from transformers import Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from utils.longseq.llama_attn_replace import replace_llama_attn
from peft import LoraConfig, get_peft_model

from datasets import load_from_disk

import numpy as np
from tqdm import tqdm

from models.llama_flash_attn.modeling_llama import LlamaForCausalLM

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

import os
os.environ["WANDB_PROJECT"]="longlora"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    data_dir: Optional[str] = field(default="./data")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    run_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    lora_rank: int = field(
        default=16,
        metadata={"help": "LoRA rank"},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"},
    )
    proof_pile_file: str = field(
        default="proof_pile.bin",
        metadata={"help": "Proof pile file"},
    )
    pg19_validation_file: str = field(
        default="pg19_validation.bin",
        metadata={"help": "PG19 validation file"},
    )
    

# hyper parameters about hyclora method
@dataclass
class HyCLoRAArguments:
    use_hyclora: bool = field(
        default=True,
        metadata={"help": "Whether to replace the original training module to fused training module"}
    )
    layer_type: str = field(
        default="baseline",
        metadata={"help": "Different types of fused training layers (Available: [baseline | intra | intra_inter])"}
    )
    iteration_threshold: int = field(
        default=5,
        metadata={"help": "calibration steps"}
    )
    softmax_outlier_ratio: float = field(
        default=0.05,
        metadata={"help": "softmax outlier selection ratio"}
    )
    layernorm_outlier_ratio: float = field(
        default=0.005,
        metadata={"help": "layernorm outlier channels selection ratio"}  
    )
    q_bit: int = field(
        default=4,
        metadata={"help": "quantization bit"}
    )
    

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def tokenize_fn(tokenizer, example):
    context_length = tokenizer.model_max_length
    outputs = tokenizer(
        tokenizer.eos_token.join(example["text"]),
        truncation=False,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}


def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256):
    all_ix = list(range(0, len(data) - seq_length, sliding_window))
    all_ix.pop()

    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx+batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y


def iceildiv(x, y):
    return (x + y - 1) // y


def evaluate(model, data, batch_size, device, seq_length, sliding_window=256, use_cache=False):
    stats = {}

    model.eval()

    loss_list_val, acc_list = [], []
    loss_step_list_val = []

    with torch.no_grad():
        print(f"Using seq length {seq_length}")
        torch.set_printoptions(sci_mode=False)
        for idx, (x, y) in tqdm(
            enumerate(
                get_as_batch(
                    data['val'], 
                    seq_length, 
                    batch_size, 
                    device=device,
                    sliding_window=sliding_window
                )
            ),
            total=iceildiv(
                iceildiv(len(data['val']), sliding_window),
                batch_size
            )
        ):
            val_loss = 0.
            acc = 0.
            cnt = 0

            for part_idx, i in enumerate(range(0, x.shape[1], seq_length)):
                part_len = x[:, i:i + seq_length].shape[1]

                outputs = model(
                    input_ids=x[:, i:i + seq_length],
                    labels=x[:, i:i+seq_length].contiguous(),
                    use_cache=use_cache)

                val_loss = outputs.loss * part_len + val_loss
                acc = ((outputs.logits.argmax(-1) == y[:, i:i+seq_length]).float().sum()) + acc
                cnt += part_len
                while len(loss_step_list_val) <= part_idx:
                    loss_step_list_val.append([])
                loss_step_list_val[part_idx].append(outputs.loss.item())
            val_loss /= cnt
            acc /= cnt
            
            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())
            
            if idx % 50 == 0:
                val_loss = torch.as_tensor(loss_list_val).mean().item()
                print(f"Validation loss: {val_loss}")
                print(f"Validation perplexity: {2.71828 ** val_loss}")

    stats['val_acc'] = torch.as_tensor(acc_list).mean().item()
    stats['val_loss'] = torch.as_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    stats['val_perplexity_per_chunk'] = torch.exp(torch.as_tensor(loss_step_list_val).mean(dim=1))

    return stats


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments, HyCLoRAArguments))
    model_args, training_args, hyclora_args = parser.parse_args_into_dataclasses()

    # NOTE: May expand supported model types in the future
    if model_args.model_type == "llama":
        replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)
    else:
        raise ValueError(f"Unsupported model type: {model_args.model_type}")

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}

    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    model.set_fused_llama_layer(hyclora_args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    dataset = load_from_disk(training_args.data_dir)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    config = LoraConfig(
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        target_modules=targets,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    # enable trainable params
    [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]
        
    # replace model
    print(model)

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    # model.gradient_checkpointing_enable()  # enable gradient checkpointing
    
    for name, module in model.named_modules():
        module.name = name
    model = model.to(torch.bfloat16)
    
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=None,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    
    data = {'val': np.memmap(training_args.proof_pile_file, dtype=np.uint16, mode='r')}
    stats = evaluate(model, data, 1, "cuda", training_args.model_max_length, sliding_window=256)
    print(stats)
    
    data = {'val': np.memmap(training_args.pg19_validation_file, dtype=np.uint16, mode='r')}
    stats = evaluate(model, data, 1, "cuda", training_args.model_max_length, sliding_window=256)
    print(stats)


if __name__ == "__main__":
    train()
