import os
import peft
import math
import torch
import wandb
import random
import logging
import argparse

from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from peft import (
    LoraConfig,
    PeftModelForSequenceClassification,
)
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from models.roberta.modeling_roberta import RobertaForSequenceClassification

logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

metric_key = {
    "mrpc": "f1",
    "sst2": "accuracy",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "qqp": "f1",
    "stsb": "pearsonr",
    "cola": "matthews_correlation",
    "wnli": "accuracy",
}


def parse_args():
    # type 1: simple training config
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation-file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use-slow-tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay to use."
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=1.0,
        help="Maximum norm of gradient.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num-warmup-steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub-token", type=str, help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--layer-num", type=int, default=24, help="Number of Bert layers"
    )
    parser.add_argument("--hidden-size", type=int, default=1024, help="hidden size")
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=4096,
        help="customize intermediate size",
    )
    parser.add_argument(
        "--ckpt", action="store_true", help="enable gradient checkpoint"
    )
    parser.add_argument("--r", action="store_true", help="lora rank", default=16)

    # hyclora parameters
    parser.add_argument("--use-hyclora", type=bool, default=True, help="use hyclora")
    parser.add_argument("--layer-type", type=str, default="baseline", help="layer type")
    parser.add_argument("--iteration-threshold", type=int, default=5, help="iteration threshold")
    parser.add_argument("--softmax-outlier-ratio", type=float, default=0.05, help="softmax outlier ratio")
    parser.add_argument("--layernorm-outlier-ratio", type=float, default=0.005, help="layernorm outlier ratio")
    parser.add_argument("--q-bit", type=int, default=4, help="quantization bit")
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(device_placement=False)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Handle the repository creation

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset("./utils/glue/glue.py", args.task_name)

    # Labels
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )

    config = AutoConfig.from_pretrained(
        args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name
    )
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["classifier"],
        ),
        ignore_mismatched_sizes=True,
        device_map="auto",
    )
    model.set_fused_roberta_layer(args)
    use_gradient_checkpointing = args.ckpt
    model = peft.prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=use_gradient_checkpointing
    )

    if args.ckpt:
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=args.r,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["dense", "key", "value", "query"],
    )
    model = PeftModelForSequenceClassification(model, peft_config)

    # print trainable parameter ratio:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params
    print(
        f"Total params: {total_params}, Trainable params: {trainable_params}, Trainable ratio: {trainable_ratio}"
    )

    for name, module in model.named_modules():
        module.name = name

    print(model)

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {
                i: label_name_to_id[label_list[i]] for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *texts, padding=padding, max_length=args.max_length, truncation=True
        )

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    train_max_length = 0
    dev_max_length = 0
    for item in train_dataset:
        if len(item["input_ids"]) > train_max_length:
            train_max_length = len(item["input_ids"])
    for item in eval_dataset:
        if len(item["input_ids"]) > dev_max_length:
            dev_max_length = len(item["input_ids"])
    logger.info("Train max length: %d" % train_max_length)
    logger.info("Dev max length: %d" % dev_max_length)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    metric = load_metric("glue", args.task_name)
    
    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    train_step = 0
    completed_steps = 0
    best_metric = 0
    model.to(args.device)
    model = model.to(torch.bfloat16)

    # begin training
    for epoch in range(args.num_train_epochs):
        model.train()
        for _, batch in enumerate(train_dataloader):
            train_step += 1

            for k, v in batch.items():
                batch[k] = v.to(args.device)

            outputs = model(**batch)

            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps

            optimizer.zero_grad()
            loss.backward()
            torch.utils.checkpoint.first_iter = False

            torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), args.max_gradient_norm
            )
            optimizer.step()  #! optimizer generate there
            lr_scheduler.step()
            completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        with torch.no_grad():
            model.eval()
            for _, batch in enumerate(eval_dataloader):
                for k, v in batch.items():
                    batch[k] = v.to(args.device)
                outputs = model(**batch)
                predictions = (
                    outputs.logits.argmax(dim=-1)
                    if not is_regression
                    else outputs.logits.squeeze()
                )
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )

        eval_metric = metric.compute()
        print(f"epoch {epoch}: {eval_metric}")

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        with open(os.path.join(args.output_dir, "result.txt"), "a") as f:
            f.write(
                "lr:%f, bsz:%d, result:%f\n"
                % (
                    args.learning_rate,
                    args.per_device_train_batch_size,
                    best_metric,
                )
            )


if __name__ == "__main__":
    main()