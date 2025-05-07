import torch
from torch.nn.functional import softmax

# Customize label map to match your model's
label_map = {
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the input sentence
    tokens = tokenizer(sentence, return_tensors="pt", is_split_into_words=False)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)

    # Convert token IDs to tokens
    tokens_list = tokenizer.convert_ids_to_tokens(input_ids[0])
    pred_labels = [label_map[p.item()] for p in predictions[0]]

    # Filter out special tokens like [CLS], [SEP]
    result = []
    for token, label in zip(tokens_list, pred_labels):
        if token not in tokenizer.all_special_tokens:
            result.append((token, label))
    return result



#def tokenize_and_predict(sentence, tokenizer, model):
#    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, is_split_into_words=False)
#    with torch.no_grad():
#        outputs = model(**inputs)
#    logits = outputs.logits
#    predictions = torch.argmax(logits, dim=2)
#    predicted_labels = [id2label[label_id.item()] for label_id in predictions[0]]
#    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
#    return tokens, predicted_labels
#
