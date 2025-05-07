import torch
from torch.nn.functional import softmax

# Customize label map to match your model's
id2label = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-LOC",
    4: "I-LOC",
    5: "B-ORG",
    6: "I-ORG",
    7: "B-MISC",
    8: "I-MISC"
}

def tokenize_and_predict(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, is_split_into_words=False)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    predicted_labels = [id2label[label_id.item()] for label_id in predictions[0]]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, predicted_labels

