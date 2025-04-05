from transformers import BartForConditionalGeneration, BartTokenizer

class RAGGenerator:
    def __init__(self,device="cuda"):
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
        

if __name__=="__main__":
    generator=RAGGenerator()


