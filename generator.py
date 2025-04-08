from typing import List
from transformers import BartForConditionalGeneration, BartTokenizer


class RAGGenerator:
    def __init__(self, device="cuda", model="facebook/bart-base"):
        self.device = device
        self.tokenizer:BartTokenizer = BartTokenizer.from_pretrained(model)
        self.model:BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(model).to(device)

    def tokenize(self,
            questions:List[str],
            contexts: List[List[str]],
            answers:List[str],
            input_max_length=1024,
            target_max_length=128,
        ):
        context_texts=[]
        for q,c in zip(questions,contexts):
            context_texts.append("</s>".join([q]+c))
        tokenized= self.tokenizer(
            text=context_texts,
            text_target=answers,
            max_length=input_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            device=self.device,
        )
        # input_ids: [N x L]
        # labels: [N x L]
        # attention_mask: [N x L]
        tokenized["labels"]=tokenized["labels"][:,:target_max_length].contiguous()
        return tokenized  

    def get_loss(
        self,
        questions:List[str],
        contexts:List[List[str]],
        answers:List[str],
        input_max_length=1024,
        target_max_length=128
    ):
        batch_input=self.tokenize(questions,contexts,answers,input_max_length,target_max_length)
        input_ids=batch_input["input_ids"]
        attn_mask=batch_input["attention_mask"]
        labels=batch_input["labels"]
        ret=self.model.forward(input_ids=input_ids,attention_mask=attn_mask,labels=labels)
        # [loss,logits,encoder_last_hidden_state]
        return ret["loss"]

if __name__ == "__main__":
    generator = RAGGenerator(device="cpu")
