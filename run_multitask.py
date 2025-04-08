import os
import sys
import json
import warnings
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

import torch
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    AutoConfig,
)
from typing import Dict
from collections import defaultdict
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers.utils import send_example_telemetry
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel, get_peft_model, TaskType, LoraConfig
from copy import deepcopy
from tqdm import tqdm

from utils.math_10k import compute_metrics

from models.llama.modeling_llama import LlamaForCausalLM
from models.mistral.modeling_mistral import MistralForCausalLM

os.environ["WANDB_PROJECT"] = "hyclora-math10k"

# training template

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

no_header_prompt_template = """\
### Instruction:
%s

### Response:
"""

alpaca_prompt_template = """Below is an instruction that \
describes a task, paired with an input that provides \
further context. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

alpaca_prompt_no_input_template = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""


logger = logging.getLogger(__name__)


task_config = compute_metrics.task_config


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    attn_implementation: str = field(
        default="eager",
        metadata={"help": "Choose from [eager, sdpa, flash_attention_2]"},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    init_lora_weights: str = field(
        default='qlora',
        metadata={"help": "Choose from [qlora, qpissa]"},
    )
    rank: int = field(
        default=16,
        metadata={"help": "Rank of LoRA adapters. LoftQ does not require this config. Used for fp16 LoRA or QLoRA."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    backbone_type: str = field(
        default="fp16",
        metadata={"help": "backbone precision (16bit or nf4)"},
    )
    weight_bit: int = field(
        default=4,
        metadata={"help": "Weight quantization bit."},
    )
    activation_bit: int = field(
        default=4,
        metadata={"help": "Activation quantization bit."},
    )
    gradient_bit: int = field(
        default=4,
        metadata={"help": "Gradient quantization bit."},
    )
    target_module_type: str = field(
        default="fused_lora_module",
        metadata={"help": "Type of target module. Options: fused_lora_module, fused_lora_module_loftq, fused_lora_module_pissa, fused_lora_module_migrate"},
    )
    enable_replace: bool = field(
        default=True,
        metadata={"help": "enable the replace"},
    )
    mask_a: bool = field(
        default=False,
        metadata={"help": "mask a"},
    )
    mask_b: bool = field(
        default=False,
        metadata={"help": "mask b"},
    )
    weight_quantization_type: str = field(
        default="per-channel",
        metadata={"help": "quantization config"},
    )
    activation_quantization_type: str = field(
        default="per-channel",
        metadata={"help": "quantization config"},
    )
    split_activation_precision: bool = field(   
        default=False,
        metadata={"help": "split activation precision"},
    )


@dataclass
class DataTrainingArguments:
    task: Optional[str] = field(default=None)
    data_dir: Optional[str] = field(default=None)
    train_dataset: Optional[str] = field(default=None)
    eval_dataset: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    test_split: Optional[str] = field(default="validation")
    train_on_inputs: bool = field(default=False)
    max_length: Optional[int] = field(default=512)
    use_normalized_template: bool = field(default=False)
    temperature: Optional[float] = field(default=None)
    top_p: Optional[float] = field(default=None)
    top_k: Optional[float] = field(default=None)
    greedy_decoding: bool = field(default=False)
    
    
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
    
    
# supervised dataset
class SupervisedDataset(Dataset):
    def __init__(
        self, 
        task: str, 
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_split="train", 
        dataset=None, 
        seed=42, 
        max_n_example=None,
        **kwargs,
    ):
        super(SupervisedDataset, self).__init__()
        result = defaultdict(list)
        self.raw_dataset, self.trigger_tokens, self.num_labels = None, None, None
        
        dataset_config = task_config[task]
        task_prompt_template = dataset_config["task_prompt_template"]
        trigger_tokens = dataset_config["trigger_tokens"]
        self.trigger_tokens = trigger_tokens

        if dataset is None:
            print("loading data for dataset: ", data_path)
            task_dataset = load_dataset("json", data_files=os.path.join(data_path, f"{data_split}.json"))["train"]
        if max_n_example is not None:
            task_dataset = task_dataset.shuffle(seed=seed)
            task_dataset = task_dataset.select(range(max_n_example))

        # save raw_dataset pointer for access raw strings
        self.raw_dataset = task_dataset if data_split != "train" else None

        # tokenize and intervene
        for i, data_item in enumerate(tqdm(task_dataset)):
            # set up prompt
            if task == "commonsense":
                base_prompt = task_prompt_template % (data_item['instruction'])
                base_input = base_prompt + trigger_tokens + data_item["answer"] + tokenizer.eos_token
            elif task == "math": # we strip since these are model generated examples.
                base_prompt = task_prompt_template % (data_item['instruction'])
                base_input = base_prompt + data_item["output"] + tokenizer.eos_token
            else:
                raise ValueError(f"Unrecognized task: {task}")
            
            PADDING_LENGTH = 32
            # tokenize
            base_prompt_ids = tokenizer(
                base_prompt, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
            base_prompt_length = len(base_prompt_ids)
            if data_split == "train":
                base_input_ids = tokenizer(
                    base_input, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
                
                # pad base_input_ids to the times of PADDING_LENGTH
                if len(base_input_ids) % PADDING_LENGTH != (PADDING_LENGTH - 1):
                    padding_ids = torch.tensor([tokenizer.pad_token_id] * ((PADDING_LENGTH - 1) - len(base_input_ids) % PADDING_LENGTH))
                    base_input_ids = torch.cat((base_input_ids, padding_ids))
                
                output_ids = deepcopy(base_input_ids)
                output_ids[:base_prompt_length] = IGNORE_INDEX
                    
                result["input_ids"].append(base_input_ids)
                result["labels"].append(output_ids)
            else:
                result["input_ids"].append(base_prompt_ids)
                
            result["id"].append(i)
            
            # add a single padding token BEFORE input_ids and fix everything
            result["input_ids"][-1] = torch.cat((torch.tensor([tokenizer.pad_token_id,]), result["input_ids"][-1]))
            if data_split == "train":
                result["labels"][-1] = torch.cat((torch.tensor([IGNORE_INDEX]), result["labels"][-1]))
            result["attention_mask"].append((result["input_ids"][-1] != tokenizer.pad_token_id).int())

        self.input_ids = result["input_ids"]
        self.attention_mask = result["attention_mask"]
        self.labels = result["labels"] if "labels" in result else None
        self.id = result["id"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.labels is not None:
            return dict(
                input_ids=self.input_ids[i],
                attention_mask=self.attention_mask[i],
                labels=self.labels[i],
            )
        else:
            return dict(
                input_ids=self.input_ids[i],
                attention_mask=self.attention_mask[i],
                id=self.id[i],
            )
            

def init(model):
    for name, module in model.named_modules():
        if name.split('.')[-1] in ['q_proj', 'k_proj', 'v_proj', 'up_proj', 'gate_proj', 'o_proj', 'down_proj']:
            module.initialize()


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, HyCLoRAArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, hyclora_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, hyclora_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    assert data_args.task in {"commonsense", "math"}
    assert data_args.task in task_config, f"Unrecognized task: {data_args.task}"

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_translation", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, float16 training: {training_args.fp16}, "
        + f"bfloat16 training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        model_max_length=data_args.max_length,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.unk_token is None: # For Llama-3
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
    else:
        tokenizer.pad_token = tokenizer.unk_token

    # Load dataset
    train_datasets = task_config[data_args.task]["train_datasets"] if data_args.train_dataset is None else [data_args.train_dataset]
    eval_datasets = task_config[data_args.task]["eval_datasets"] if data_args.eval_dataset is None else [data_args.eval_dataset]

    train_dataset = SupervisedDataset(
        data_args.task, 
        os.path.join(data_args.data_dir, train_datasets[0]) if data_args.data_dir is not None else train_datasets[0], 
        tokenizer, 
        data_split="train", 
        seed=training_args.seed, 
        max_n_example=data_args.max_train_samples,
    )
    trigger_tokens = train_dataset.trigger_tokens

    all_eval_datasets = {}
    for eval_dataset in eval_datasets:
        test_splits = data_args.test_split.split(";")
        all_eval_datasets[eval_dataset] = {}
        for split in test_splits:
            raw_eval = SupervisedDataset(
                data_args.task, 
                os.path.join(data_args.data_dir, eval_dataset), 
                tokenizer, 
                data_split=split, 
                seed=training_args.seed, 
                max_n_example=data_args.max_eval_samples,
            )
            all_eval_datasets[eval_dataset][split] = [raw_eval, raw_eval.raw_dataset]
    eval_datasets = all_eval_datasets

    # Load model
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, 
        attn_implementation=model_args.attn_implementation
    )
    
    if 'llama' in model_args.model_name_or_path:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            attn_implementation="eager"
        )
        model.set_fused_llama_layer(hyclora_args)
    elif 'mistral' in model_args.model_name_or_path:
        model = MistralForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            use_cache=False if model_args.gradient_checkpointing_enable else True,
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            attn_implementation="eager"
        )
        model.set_fused_mistral_layer(hyclora_args)

    if training_args.gradient_checkpointing:
        logger.info("Use gradient checkpointing with LoRA.")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()

    # PEFT
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "gate_proj", "o_proj", "down_proj"]
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.0,
            target_modules=target_modules,
            init_lora_weights=True if model_args.init_lora_weights == 'qlora' else model_args.init_lora_weights
        )
        model = get_peft_model(model, lora_config)
    elif model_args.adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(
            model,
            model_args.adapter_name_or_path,
            is_trainable=True,
            token=model_args.token,
        )
    else:
        print("Using LoftQ initialization.")
        model = PeftModel.from_pretrained(
            model,
            model_args.model_name_or_path,
            subfolder='loftq_init',
            is_trainable=True,
            token=model_args.token,
        )
    logger.info(model)
    logger.info(model.print_trainable_parameters())

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics=None,
    )

    model = model.to(torch.bfloat16)
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics = train_result.metrics
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        model.eval()
        assert next(model.parameters()).is_cuda

        eval_results = {}
        for dataset_name in eval_datasets:
            # split evalset into chunks
            for split, (eval_dataset, data_items) in eval_datasets[dataset_name].items():
                generations, stats = compute_metrics.compute_metrics(
                    data_args.task, 
                    dataset_name, 
                    model, 
                    tokenizer, 
                    eval_dataset, 
                    data_items,
                    trigger_tokens, 
                    None, 
                    training_args.per_device_eval_batch_size, 
                    None,
                    split, 
                    data_args.greedy_decoding, 
                    data_args.temperature,
                    data_args.top_p, 
                    data_args.top_k
                )

                # log
                eval_results.update(stats)
                generations = stats if generations is None else generations
                result_json_file_name = f"{training_args.output_dir}/{dataset_name}_{split}_outputs.json"
                with open(result_json_file_name, 'w') as json_file:
                    json.dump(generations, json_file, indent=4)

        # log final eval stats
        eval_results["mean"] = np.mean(list(eval_results.values()))
        result_json_file_name = f"{training_args.output_dir}/eval_results.json"
        with open(result_json_file_name, 'w') as json_file:
            json.dump(eval_results, json_file, indent=4)

if __name__ == "__main__":
    main()