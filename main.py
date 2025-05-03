from generator import RAGGenerator
from dataset import NQOpenDataset, NQOpenDatasetFactory
from tqdm import tqdm
from logger import TBWriter
from torch.utils.data import DataLoader
import torch
from typing import List
from comprehensive_encoder import ComprehensiveEncoder

DEVICE = "cuda"
N_DOCS=4

def generate(
    question:str,
    passages:List[str],
    encoders:ComprehensiveEncoder,
    generator:RAGGenerator,
):
    chunked_passages=[]
    for p in passages:
        words=p.split(" ")
        for k in range(0,len(words),50):
            chunked_passages.append(" ".join(words[k:min(k+50,len(words))]))
    question_embeddimg=encoders.encode_questions([question])  # [1 x emb_dim]
    context_embedding=encoders.encode_paragraphs([chunked_passages])  # [N x emb_dim]
    relevance=(question_embeddimg*context_embedding).sum(dim=-1)  # [N]
    _,relevant_indices=torch.topk(relevance,k=4)
    relevant_indices=relevant_indices.cpu().tolist()
    context= [chunked_passages[i] for i in relevant_indices]
    results=generator.rag_injected_generate([question],[context])
    results=generator.decode_answers(results)
    return results[0]


def main():
    generator = RAGGenerator(DEVICE)
    generator.load_state_dict(torch.load("ckpt/train_generator_weighted/2025-04-21_10:12:14/generator-200.pt",map_location=DEVICE))
    generator.eval()
    dataset_factory = NQOpenDatasetFactory(device=DEVICE)
    generator.model.config.n_docs=N_DOCS
    testset=dataset_factory.get_testset(n_docs=N_DOCS)
    testloader=DataLoader(testset,batch_size=64,shuffle=False,collate_fn=testset.collate_fn)
    with torch.no_grad():
        for x in testloader:
            questions=x["question"]
            contexts=x["context"]
            results=generator.rag_injected_generate(questions,contexts)
            results=generator.decode_answers(results)
            for i,(q,c,r,a) in enumerate(zip(questions,contexts,results,x["answer"])):
                print(f"question {i}: {q}")
                for j,cc in enumerate(c):
                    print(f"context {i}-{j}: {cc}")
                print(f"answer {i}: {r}")
                print(f"correct answer {i}: {a}")

    

if __name__ == "__main__":
    main()
