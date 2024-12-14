import torch
from transformers import AutoTokenizer, AutoModel
import os

import numpy as np
from transformers import TrainingArguments, Trainer, TrainerCallback
from tqdm import tqdm, trange
from scipy.stats import spearmanr
import torch.nn.functional as F
from datasets import load_dataset

from matplotlib import pyplot as plt
import wandb
import random

from mteb_eval import MTEB

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-base")

EVAL_BATCH_SIZE = 32
OUTPUT_DIR = "test_trainer_v4"


def change_dropout(model, p):
    model.embeddings.dropout.p = p
    for layer in model.encoder.layer:
        layer.attention.self.dropout.p = p
        layer.attention.output.dropout.p = p
        layer.output.dropout.p = p
    return model


def model_init(trial):
    model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-roberta-base").to(
        "cuda"
    )
    return model


def sweep_config(trial):
    return {
        "method": "random",
        "metric": {"name": "main_score", "target": "main_score", "goal": "maximize"},
        "parameters": {
            "training_max_length": {"values": [32, 64, 128]},
            "per_device_train_batch_size": {"values": [16, 32, 64]},
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-6,
                "max": 3e-5,
            },
            "warmup_steps": {"distribution": "int_uniform", "min": 0, "max": 1000},
            "model_dropout": {"values": [0.01, 0.05, 0.1, 0.15, 0.2]},
            "temperature": {"values": [0, 0.05, 0.1, 0.2]},
        },
        "early_terminate": {"type": "hyperband", "min_iter": 1, "eta": 2},
    }


# import dataset
dataset = (
    load_dataset(
        "abokbot/wikipedia-first-paragraph",
        split="train",
        trust_remote_code=True,
    )
    .shuffle(seed=42)
    .select(range(256000))
)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


train_dataset = dataset.map(tokenize_function, batched=True)


class SentenceEmbedder:
    def __init__(self, model, tokenizer, dim=768):
        torch.cuda.empty_cache()
        self.model = model
        self.tokenizer = tokenizer
        self.dim = dim

    def encode(
        self, sentences, max_length=512, batch_size=32, convert_to_tensor=True, **kwargs
    ):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = inputs["input_ids"].to("cuda")
            attention_mask = inputs["attention_mask"].to("cuda")

            embeddings = torch.zeros(len(sentences), self.dim).to("cuda")
            for i in trange(0, len(sentences), batch_size):
                batch_input_ids = input_ids[i : i + batch_size]
                batch_attention_mask = attention_mask[i : i + batch_size]
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                )
                embeddings[i : i + batch_size] = outputs.pooler_output
            if convert_to_tensor:
                return embeddings
            else:
                return embeddings.cpu().numpy()


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    lr_scheduler_type="linear",
    report_to="wandb",
    save_steps=3000,
    num_train_epochs=1,
    seed=1,
    eval_strategy="steps",
    eval_steps=2000,
    metric_for_best_model="main_score",
    greater_is_better=True,
)


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, score_threshold=0.9):
        self.score_threshold = score_threshold

    def on_evaluate(self, args, state, control, **kwargs):
        score = kwargs["metrics"]["main_score"]
        if score < self.score_threshold:
            return {"should_training_stop": True}
        else:
            return {}


training_args.eval_delay = 0

"""
Evaluation
Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere” by Tongzhou Wang and Phillip Isola, ICML 2020.
"""

loss_func = torch.nn.CrossEntropyLoss()


retrieval_task_list = [
    "LEMBSummScreenFDRetrieval",
    "LEMBQMSumRetrieval",
    "LEMBWikimQARetrieval",
    "LEMBNarrativeQARetrieval",
]


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs
    ):
        print(trial)
        self.args.eval_steps = (
            64000 // trial["per_device_train_batch_size"]
        )  # eval every 64000 samples
        print("Eval steps: ", self.args.eval_steps)
        model_dropout = trial["model_dropout"]
        if model_dropout:
            self.model = change_dropout(self.model, model_dropout)
        super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=1,
    ):
        # temperature = 0.05
        temperature = self.args.temperature
        self.model.train()

        batch_size = self.args.per_device_train_batch_size
        mini_batch_size = 16

        training_max_length = self.args.training_max_length
        inputs["input_ids"] = inputs["input_ids"][:, :training_max_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :training_max_length]

        output1 = torch.zeros(batch_size, 768).to("cuda")
        output2 = torch.zeros(batch_size, 768).to("cuda")

        for i in range(0, batch_size, mini_batch_size):
            next_idx = min(i + batch_size, batch_size)

            input1 = inputs["input_ids"][i:next_idx]
            input2 = inputs["input_ids"][i:next_idx]

            mask1 = inputs["attention_mask"][i:next_idx]
            mask2 = inputs["attention_mask"][i:next_idx]

            output1[i:next_idx] = model(input1, attention_mask=mask1).pooler_output
            output2[i:next_idx] = model(input2, attention_mask=mask2).pooler_output

        if temperature == 0:
            M = output1 @ (output2.T)
        else:
            output1 = F.normalize(output1, p=2, dim=1)
            output2 = F.normalize(output2, p=2, dim=1)

            M = output1 @ (output2.T)
            M /= temperature

        # Compute log softmax along the rows
        labels = torch.arange(0, output1.size(0), device=M.device, dtype=torch.long)
        loss = loss_func(M, labels)

        self.log(
            {
                "loss": loss.item(),
            }
        )
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Evaluate model performance on STS benchmark using the "all" setting

        Args:
            model: The transformer model to evaluate
            version (int): STS version (12-16)
            batch_size (int): Batch size for processing
        Returns:
            tuple: (overall correlation percentage, dict of domain-wise correlations)
        """
        evaluation = MTEB(tasks=retrieval_task_list)

        sentence_embedder = SentenceEmbedder(self.model, tokenizer)
        results = evaluation.run(
            sentence_embedder,
            output_folder="output_dir",
            overwrite_results=True,
            batch_size=EVAL_BATCH_SIZE,
            verbosity=0,
        )
        results = results[0].scores["test"][0]

        output = {}
        # convert all values to float
        for key in results:
            try:
                value = float(results[key])
                output[key] = value
            except:
                pass
        output["eval_loss"] = output["main_score"]
        self.log(output)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, results
        )
        self._memory_tracker.stop_and_update_metrics(results)

        return results


def compute_objective(metrics):
    return metrics["main_score"]


import gc

gc.collect()
torch.cuda.empty_cache()

trainer = MyTrainer(
    model=None,
    args=training_args,
    model_init=model_init,
    train_dataset=train_dataset,
    eval_dataset=[None],
    # callbacks=[EarlyStoppingCallback()],
)

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="wandb",
    hp_space=sweep_config,
    n_trials=100,
    compute_objective=compute_objective,
)
