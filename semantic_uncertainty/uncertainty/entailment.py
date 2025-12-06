import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EntailmentDeberta:
    """
    Wrapper around a DeBERTa-v3 MNLI model for NLI / textual entailment.

    Uses labels (standard MNLI convention):
        0 = contradiction
        1 = neutral
        2 = entailment
    """

    def __init__(
        self,
        model_name: str = "mrm8488/deberta-v3-large-finetuned-mnli",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[EntailmentDeberta] Loading model `{model_name}` on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def check_implication(self, premise: str, hypothesis: str) -> int:
        """
        Returns the predicted label id:
            0 = contradiction
            1 = neutral
            2 = entailment
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits
        label_id = int(torch.argmax(logits, dim=-1).item())
        return label_id