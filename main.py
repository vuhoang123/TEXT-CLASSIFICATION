import os
import torch
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")


class PhobertForNewsClassification(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.phobert = AutoModelForSequenceClassification.from_pretrained(
            "vinai/phobert-base", num_labels=num_classes)
        for param in self.phobert.roberta.parameters():
            param.requires_grad = False

    def forward(self, input_ids, labels):
        out = self.phobert(input_ids, labels=labels)
        probs = torch.softmax(out['logits'], dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().detach().numpy()
        out['preds'] = preds
        return out


category2id = {'Đời sống': 0,
               'Khoa học': 1,
               'Kinh doanh': 2,
               'Pháp luật': 3,
               'Sức Khoẻ': 4,
               'Thế giới': 5,
               'Thể thao': 6,
               'Văn hoá': 7,
               'Vi tính': 8,
               'Chính trị xã hội': 9}
model = PhobertForNewsClassification(len(category2id))
model.load_state_dict(torch.load(
    'model/model.pt', map_location=torch.device('cpu')))
device = "cpu"


def predict_category(text):
    # Tokenize input text and add special tokens
    encoded_input = tokenizer.encode_plus(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Make prediction
    input_ids = encoded_input["input_ids"]
    labels = None

    # Get predicted label
    with torch.no_grad():
        output = model(input_ids, labels)
    logits = output['logits']
    probs = torch.softmax(logits, dim=-1)
    pred_label_id = torch.argmax(probs, dim=-1).item()
    label_probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    result = []
    result.append(list(category2id.keys())[list(category2id.values()).index(pred_label_id)])
    for label, probability in zip(category2id.keys(), label_probabilities):
        result.append(f"{label}: {probability:.4f}")

    return result
