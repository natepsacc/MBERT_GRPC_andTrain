import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "ModernBERT-domain-classifier"
BASE_MODEL = "answerdotai/ModernBERT-base"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


def predict(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    predicted_id = probs.argmax().item()
    return {
        "label": model.config.id2label[predicted_id],
        "confidence": round(probs[predicted_id].item(), 4),
        "scores": {
            model.config.id2label[i]: round(p.item(), 4)
            for i, p in enumerate(probs)
        },
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = predict(text)
        print(f"Label:      {result['label']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print("Scores:")
        for label, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
            print(f"  {label:<20} {score:.1%}")
    else:
        # Interactive mode
        print("CSR Request Classifier — enter text to classify (Ctrl+C to quit)\n")
        while True:
            try:
                text = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not text:
                continue
            result = predict(text)
            print(f"  => {result['label']} ({result['confidence']:.1%})\n")
