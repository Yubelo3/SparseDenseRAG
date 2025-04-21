from torch.utils.data.dataset import Dataset
import jsonlines
import pickle
from sparse_retriever import SparseRetriever
from comprehensive_encoder import ComprehensiveEncoder
from tqdm import tqdm
import os
import torch
import pandas as pd


DEVICE="cuda"

def main():
    # gather_contexts()
    preprocess()

def gather_contexts(
    trainset_size=20000,
    validset_size=2000,
    testset_size=2000,
):
    context_paths=[
        "data/hotpotqa/train_context.jsonl",
        "data/hotpotqa/valid_context.jsonl",
        "data/hotpotqa/test_context.jsonl"
    ]
    parquet_paths=[
        "data/hotpotqa/train0.parquet",
        "data/hotpotqa/valid.parquet",
        "data/hotpotqa/test.parquet",
    ]
    dataset_sizes=[trainset_size,validset_size,testset_size]
    sparse_retriever = SparseRetriever()
    for i in range(3):
        context_path=context_paths[i]
        parquet_path=parquet_paths[i]
        dataset_size=dataset_sizes[i]
        next_idx = 0
        if os.path.exists(context_path):
            with jsonlines.open(context_path, "r") as reader:
                for line in reader:
                    next_idx = line[0]+1
        raw_data = pd.read_parquet(parquet_path,columns=["question","answer","level"]).to_dict("list")
        with jsonlines.open(context_path, "a") as writer:
            for i in tqdm(range(min(dataset_size,len(raw_data["question"])))):
                if i >= dataset_size:
                    break
                if i < next_idx:
                    continue
                search_results = sparse_retriever.query(
                    raw_data["question"][i], 3)
                writer.write([i, search_results])

def preprocess():
    parquet_paths=[
        "data/hotpotqa/train0.parquet",
        "data/hotpotqa/valid.parquet",
        "data/hotpotqa/test.parquet",
    ]
    context_paths=[
        "data/hotpotqa/train_context.jsonl",
        "data/hotpotqa/valid_context.jsonl",
        "data/hotpotqa/test_context.jsonl"
    ]
    dump_paths=[
        "data/hotpotqa/trainset.pkl",
        "data/hotpotqa/validset.pkl",
        "data/hotpotqa/testset.pkl"
    ]
    encoders = ComprehensiveEncoder(DEVICE)
    for i in range(3):
        parquet_path=parquet_paths[i]
        context_path=context_paths[i]
        dump_path=dump_paths[i]
        data = {"question": [], "context": [], "context_score": [], "answer": []}
        raw_data=pd.read_parquet(parquet_path,columns=["question","answer","level"]).to_dict("list")
        with jsonlines.open(context_path, "r") as context_reader:
            for i, context in enumerate(tqdm(context_reader)):
                assert i == context[0]
                question = raw_data["question"][i]
                question_embedding: torch.Tensor = encoders.encode_questions([
                                                                                question])
                search_results = context[1]
                splitted_search_results=[]
                for s in search_results:
                    words=s.split(" ")
                    for k in range(0,len(words),50):
                        splitted_search_results.append(" ".join(words[k:min(k+50,len(words))]))
                        if len(splitted_search_results)>5000:
                            break
                search_results=splitted_search_results
                if (len(search_results) < 5):
                    print(f"WARNING: not sufficient context for {i}")
                    print(f"question is {question}")
                    continue
                context_embedding: torch.Tensor = encoders.encode_paragraphs(
                    search_results)
                similarity=(question_embedding*context_embedding).sum(dim=-1)
                context_score, relevant_context_indices = torch.topk(
                    similarity, k=5, largest=True, sorted=True)
                
                data["question"].append(question)
                data["context_score"].append(context_score.cpu().tolist())
                data["context"].append(
                    [search_results[i] for i in relevant_context_indices.cpu().tolist()])
                data["answer"].append([raw_data["answer"][i]])
                if (i+1) % 1000 == 0:
                    partial_dump_path = f"{dump_path.split(".")[0]}-{i+1}.{dump_path.split(".")[1]}"
                    with open(partial_dump_path, "wb") as f:
                        pickle.dump(data, f)
            with open(dump_path, "wb") as f:
                pickle.dump(data, f)

if __name__=="__main__":
    # ['id', 'question', 'answer', 'type', 'level', 'supporting_facts','context']
    # data = pd.read_parquet('data/hotpotqa/test.parquet',columns=["question","answer","level"]).to_dict("list")
    main()
