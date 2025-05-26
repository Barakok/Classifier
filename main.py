import torch
from model import BertBaseClassifier
from parseData import NewsDataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    BertTokenizer,
)

newsDataset = NewsDataset("ag_news")
newsDataset.tokenize()

model = BertBaseClassifier("bert-base-multilingual-cased", 4)
bertTokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

data_collator = DataCollatorWithPadding(tokenizer=bertTokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=newsDataset.tokenized_dataset["train"],
    eval_dataset=newsDataset.tokenized_dataset["test"],
    processing_class=bertTokenizer,
    data_collator=data_collator,
)

trainer.train()

torch.save(model.state_dict(), "bert_base_classifier.pth")
