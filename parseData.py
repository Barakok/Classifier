from datasets import load_dataset
from transformers import BertTokenizer

bertTokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


def tokenize(example):
    return bertTokenizer(
        example["text"], truncation=True, padding="max_length", max_length=128
    )


class NewsDataset:

    def __init__(self, datasetName):
        self.dataset = load_dataset(datasetName)
        self.labels = self.dataset["train"].features["label"].names
        print(f"Уникальные метки: {self.labels}")
        print(f"Количество меток: {len(self.labels)}")

    def getLabels(self):
        return self.labels

    def tokenize(self):
        self.tokenized_dataset = self.dataset.map(tokenize, batched=True)
        self.tokenized_dataset = self.tokenized_dataset.rename_column("label", "labels")
        self.tokenized_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )
