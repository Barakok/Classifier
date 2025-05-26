import torch
from model import BertBaseClassifier
from transformers import BertTokenizer
import torch.nn.functional as F


modelId = "bert-base-multilingual-cased"  # путь к нужному чекпоинту

# Параметры должны совпадать с теми, что ты использовал при обучении
model = BertBaseClassifier(modelId, num_classes=4)
model.load_state_dict(torch.load("bert_base_classifier.pth"))
model.eval()  # Переводим в режим инференса

bertTokenizer = BertTokenizer.from_pretrained(modelId)

text = "Apple is looking at buying a startup in the UK."

inputs = bertTokenizer(text, return_tensors="pt", truncation=True, padding=True)

# inputs.pop("token_type_ids", None)  # Удаляем, если есть

with torch.no_grad():
    output = model(**inputs)
    logits = output["logits"]

    probs = F.softmax(logits, dim=1)
    print(probs)

    predicted_class = torch.argmax(logits, dim=1).item()

print("Predicted class:", predicted_class)
