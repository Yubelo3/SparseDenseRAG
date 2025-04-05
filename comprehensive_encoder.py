from typing import List
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer


class ComprehensiveEncoder:
    def __init__(self,device="cuda"):
        self.device=device
        self.context_encoder = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-multiset-base").to(device)
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-multiset-base")
        self.question_encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-multiset-base").to(device)
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-multiset-base")

    def encode(self, inputs, tokenizer, encoder):
        results = []
        for sentence in inputs:
            with torch.no_grad():
                tokenized = tokenizer(
                    sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                output = encoder(**tokenized).pooler_output
            results.append(output)
        results = torch.cat(results, dim=0)
        return results

    def encode_paragraphs(self, inputs: List[str]):
        return self.encode(inputs, self.context_tokenizer, self.context_encoder)

    def encode_questions(self, inputs: List[str]):
        return self.encode(inputs, self.question_tokenizer, self.question_encoder)