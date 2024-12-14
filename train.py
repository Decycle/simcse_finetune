import torch
import numpy
from transformers import AutoTokenizer, AutoModel
import os

import numpy as np
from transformers import TrainingArguments, Trainer
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

# os.environ["WANDB_PROJECT"] = "simcse-finetune-cs567-final"

model = AutoModel.from_pretrained(
    "princeton-nlp/unsup-simcse-roberta-base"
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-base")

wandb.init(project="simcse-finetune-cs567-final")
wandb.watch(model, log="all", log_freq=50)


TRAINING_MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 128
EVAL_MAX_LENGTH = 512
MINI_BATCH_SIZE = 32

# EVAL_SHORT_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 5e-6
WARMUP_STEPS = 800

EVAL_STEPSIZE = 400  # 8 minutes per evaluation
SAVE_STEPSIZE = 400  # 8 minutes per save

MODEL_DROPOUT = 0.2  # lower dropout for longer text
TEMPERATURE = 0.05  # higher temperature for longer text
OUTPUT_DIR = "test_trainer_v4"

# import dataset
dataset = load_dataset(
    "abokbot/wikipedia-first-paragraph",
    split="train",
    trust_remote_code=True,
)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=TRAINING_MAX_LENGTH,
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets.shuffle(seed=43)


class SentenceEmbedder:
    def __init__(self, model, tokenizer, dim=768):
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
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="linear",
    warmup_steps=WARMUP_STEPS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    report_to="wandb",
    save_steps=SAVE_STEPSIZE,
    num_train_epochs=1,
    seed=1,
    gradient_accumulation_steps=1,
)
training_args.set_evaluate(strategy="steps", steps=EVAL_STEPSIZE)
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

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=1,
        temperature=TEMPERATURE,
    ):
        self.model.train()

        output1 = torch.zeros(TRAIN_BATCH_SIZE, 768).to("cuda")
        output2 = torch.zeros(TRAIN_BATCH_SIZE, 768).to("cuda")

        for i in range(0, TRAIN_BATCH_SIZE, MINI_BATCH_SIZE):
            next_idx = min(i + MINI_BATCH_SIZE, TRAIN_BATCH_SIZE)

            input1 = inputs["input_ids"][i:next_idx]
            input2 = inputs["input_ids"][i:next_idx]

            mask1 = inputs["attention_mask"][i:next_idx]
            mask2 = inputs["attention_mask"][i:next_idx]

            output1[i:next_idx] = model(input1, attention_mask=mask1).pooler_output
            output2[i:next_idx] = model(input2, attention_mask=mask2).pooler_output

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

        sentence_embedder = SentenceEmbedder(model, tokenizer)
        results = evaluation.run(
            sentence_embedder,
            output_folder="output_dir",
            overwrite_results=True,
            batch_size=EVAL_BATCH_SIZE,
            verbosity=0,
        )
        results = results[0].scores["test"][0]
        self.log(results)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, results
        )
        self._memory_tracker.stop_and_update_metrics(results)

        return results

    def get_embeddings(self, input_text, batch_size=32):
        self.model.eval()

        inputs = tokenizer(
            input_text,
            padding="max_length",
            max_length=EVAL_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        input_tokens, input_masks = inputs["input_ids"], inputs["attention_mask"]

        embeddings = []
        with torch.no_grad():
            for i in trange(0, len(input_tokens), batch_size):
                embedding = self.model(
                    input_tokens[i : i + batch_size],
                    attention_mask=input_masks[i : i + batch_size],
                ).pooler_output
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding)
        return torch.vstack(embeddings)

    def get_alignment_loss(self, embeddings_1, embeddings_2):
        return (embeddings_1 - embeddings_2).norm(p=2, dim=1).pow(2).mean().item()

    def get_uniform_loss(self, embeddings):
        distance_matrix = torch.pdist(embeddings, p=2).pow(2)
        exp_kernel = torch.exp(-2 * distance_matrix)
        uniform_loss = torch.log(exp_kernel.mean())

        return uniform_loss.item()

    def get_ranking(self, embeddings_1, embeddings_2):
        similarity_matrix = embeddings_1 @ embeddings_2.T
        rank = torch.argsort(similarity_matrix, dim=1, descending=True)
        row_indices = torch.arange(rank.size(0)).to(rank.device)
        comparison = rank == row_indices.unsqueeze(1)
        positions = comparison.nonzero()[:, 1]
        return positions

    def draw_rankings(self, ranking):
        values, counts = torch.unique(ranking, return_counts=True)
        cumulative_counts = torch.cumsum(counts, dim=0)
        cdf = cumulative_counts / cumulative_counts[-1]

        plt.figure()
        plt.step(values.cpu().numpy(), cdf.cpu().numpy(), where="post")
        plt.xlabel("Ranking")
        plt.ylabel("Cumulative Probability")
        plt.xscale("log")
        plt.title("Empirical CDF of Rankings")
        plt.grid(True)


trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=[None],
)

import gc

gc.collect()
torch.cuda.empty_cache()

trainer.evaluate()
trainer.train()

# save best model
trainer.save_model("best_model")
