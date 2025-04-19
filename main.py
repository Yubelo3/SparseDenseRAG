from generator import RAGGenerator
from dataset import NQOpenDataset, NQOpenDatasetFactory
from tqdm import tqdm
from logger import TBWriter
from torch.utils.data import DataLoader
import torch

DEVICE = "cuda"
N_DOCS=1


def main():
    generator = RAGGenerator(DEVICE)
    # generator.load_state_dict(torch.load("ckpt/train_generator/2025-04-18_09:03:46/generator-10.pt",map_location=DEVICE))
    generator.eval()
    dataset_factory = NQOpenDatasetFactory(device=DEVICE)
    generator.model.config.n_docs=N_DOCS
    testset=dataset_factory.get_testset(n_docs=N_DOCS)
    testloader=DataLoader(testset,batch_size=16,shuffle=False,collate_fn=testset.collate_fn)
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
            exit()

    

if __name__ == "__main__":
    main()
