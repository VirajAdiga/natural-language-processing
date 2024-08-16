import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""
Three steps behind pipeline method
1. Preprocessing with Tokenizer
2. Going through the model
3. Postprocessing the output
"""

raw_inputs = [
        "I am not feeling great now",
        "Food was good last Sunday"
    ]

# 1. Preprocessing with Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

# 2. Going through the model
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
outputs = model(**inputs)

# 3. Postprocessing the output
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
