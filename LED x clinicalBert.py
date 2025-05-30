!pip install transformers datasets scikit-learn
!pip install datasets


import os
os.environ["WANDB_DISABLED"] = "true"

from google.colab import files

uploaded = files.upload()  # Upload your CSV when prompted

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
from datasets import Dataset

# Load data
df = pd.read_csv("LED_summaries_labeled.csv")
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization
def tokenize_function(example):
    return tokenizer(example["note"], padding="max_length", truncation=True, max_length=512)

def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions[:, 1]
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, _ = precision_recall_curve(labels, probs)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auroc": roc_auc_score(labels, probs),
        "auprc": auc(recall, precision),
    }

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

y_train = df.iloc[train_idx]["label"].values
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
weights_tensor = torch.tensor(class_weights, dtype=torch.float)

from transformers import Trainer
from torch.nn import CrossEntropyLoss
import torch

class WeightedTrainer(Trainer):
    def __init__(self, *args, weights_tensor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_tensor = weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # ✅ Added **kwargs
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fn = CrossEntropyLoss(weight=self.weights_tensor.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 5-Fold CV
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {metric: [] for metric in ["accuracy", "f1", "auroc", "auprc"]}

for fold, (train_idx, test_idx) in enumerate(kf.split(df["note"], df["label"])):
    print(f"\n🔁 Fold {fold + 1}")

    train_data = Dataset.from_pandas(df.iloc[train_idx].reset_index(drop=True)).map(tokenize_function, batched=True)
    test_data = Dataset.from_pandas(df.iloc[test_idx].reset_index(drop=True)).map(tokenize_function, batched=True)

    train_data = train_data.remove_columns(["note"])
    test_data = test_data.remove_columns(["note"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Freeze all but last layer
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=f"./results/fold_{fold}",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        eval_strategy="epoch",
        save_strategy="no",
        logging_dir="./logs",
        disable_tqdm=True,
        learning_rate=1e-5 #Try smaller value, only fine-tuning last layer
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
        weights_tensor=weights_tensor

    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"🔍 Metrics Returned: {metrics}")

    for key in results:
        metric_key = f"eval_{key}"
        if metric_key in metrics:
            results[key].append(metrics[metric_key])
        else:
            print(f"⚠️ {metric_key} not found!")

# Final Results
print("\n📊 5-Fold CV Results (mean ± std):")
for key in results:
    mean = np.mean(results[key])
    std = np.std(results[key])
    print(f"{key.upper()}: {mean:.4f} ± {std:.4f}")
