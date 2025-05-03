from torch.utils.data.dataset import Dataset
import jsonlines
import pickle
from sparse_retriever import SparseRetriever
from comprehensive_encoder import ComprehensiveEncoder
from tqdm import tqdm
import os
import torch

DEVICE="cuda"

def main():
    # gather_contexts()
    preprocess()

def gather_contexts(
    original_train_file="data/NQ-open.efficientqa.test.1.1.jsonl",
    original_test_file="data/NQ-open.efficientqa.test.1.1.jsonl",
    trainset_size=20000,
    testset_size=2000,
):
    json_paths=[
        original_train_file,
        original_test_file,
    ]
    context_paths=[
        "data/train_context.jsonl",
        "data/test_context.jsonl"
    ]
    dataset_sizes=[trainset_size,testset_size]
    for i in range(2):
        context_path=context_paths[i]
        json_path=json_paths[i]
        dataset_size=dataset_sizes[i]
        next_idx = 0
        if os.path.exists(context_path):
            with jsonlines.open(context_path, "r") as reader:
                for line in reader:
                    next_idx = line[0]+1
        sparse_retriever = SparseRetriever()
        with jsonlines.open(json_path, "r") as reader:
            with jsonlines.open(context_path, "a") as writer:
                for i, sample in enumerate(tqdm(reader)):
                    if i >= dataset_size:
                        break
                    if i < next_idx:
                        continue
                    search_results = sparse_retriever.query(
                        sample["question"], 3)
                    writer.write([i, search_results])

def preprocess(
    original_train_file="data/NQ-open.efficientqa.test.1.1.jsonl",
    original_test_file="data/NQ-open.efficientqa.test.1.1.jsonl",
):
    json_paths=[
        original_train_file,
        original_test_file,
    ]
    context_paths=[
        "data/train_context.jsonl",
        "data/test_context.jsonl"
    ]
    dump_paths=[
        "data/trainset.pkl",
        "data/testset.pkl"
    ]
    encoders = ComprehensiveEncoder(DEVICE)
    for i in range(2):
        json_path=json_paths[i]
        context_path=context_paths[i]
        dump_path=dump_paths[i]
        data = {"question": [], "question_embedding": [],
                "context": [], "context_embedding": [], "answer": []}
        with jsonlines.open(json_path, "r") as reader:
            with jsonlines.open(context_path, "r") as context_reader:
                for i, (sample, context) in enumerate(tqdm(zip(reader, context_reader))):
                    assert i == context[0]
                    question = sample["question"]
                    question_embedding: torch.Tensor = encoders.encode_questions([
                                                                                    question])
                    search_results = context[1]
                    splitted_search_results=[]
                    for s in search_results:
                        words=s.split(" ")
                        for k in range(0,len(words),100):
                            splitted_search_results.append(" ".join(words[k:min(k+100,len(words))]))
                            if len(splitted_search_results)>2000:
                                break
                    search_results=splitted_search_results
                    if (len(search_results) < 15):
                        print(f"WARNING: not sufficient context for {i}")
                        print(f"question is {sample["question"]}")
                        continue
                    context_embedding: torch.Tensor = encoders.encode_paragraphs(
                        search_results)
                    similarity=(question_embedding*context_embedding).sum(dim=-1)
                    _, relevant_context_indices = torch.topk(
                        similarity, k=15, largest=True, sorted=True)
                    
                    data["question"].append(question)
                    # data["question_embedding"].append(
                    #     question_embedding.detach().cpu().numpy())
                    data["context"].append(
                        [search_results[i] for i in relevant_context_indices.cpu().tolist()])
                    # data["context_embedding"].append(
                    #     context_embedding[relevant_context_indices].cpu().numpy())

                    if "answer_and_def_correct_predictions" in sample:
                        answers = sample["answer_and_def_correct_predictions"]
                    else:
                        answers = sample["answer"]
                    data["answer"].append(answers)
                    if (i+1) % 1000 == 0:
                        partial_dump_path = f"{dump_path.split(".")[0]}-{i+1}.{dump_path.split(".")[1]}"
                        with open(partial_dump_path, "wb") as f:
                            pickle.dump(data, f)
        with open(dump_path, "wb") as f:
            pickle.dump(data, f)

if __name__=="__main__":
    main()