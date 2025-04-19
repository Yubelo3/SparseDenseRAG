from torch.utils.data.dataset import Dataset
import jsonlines
import pickle
from sparse_retriever import SparseRetriever
from comprehensive_encoder import ComprehensiveEncoder
from tqdm import tqdm
import os
import torch


class NQOpenDataset(Dataset):
    # require a dict [
    # "question": List[str]
    # "context": List[List[str]]
    # "answer": List[str]
    def __init__(self, dataset, index_start=0, index_end=-1, train=True, n_docs=1) -> None:
        super().__init__()
        self.n_docs=n_docs
        self.dataset = dataset
        self.train = train
        self.index_start = index_start
        self.index_end = index_end
        self.len = len(self.dataset["question"]
                       [self.index_start:self.index_end])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ret = {
            "question": self.dataset["question"][index+self.index_start],
            "context": self.dataset["context"][index+self.index_start][:self.n_docs],
            "answer": self.dataset["answer"][index+self.index_start][0],
            # "doc_score": torch.linspace(1.0,0.0,self.n_docs)*0.5+0.3
        }
        if not self.train:
            ret["answer"] = self.dataset["answer"][index+self.index_start][0]
        return ret

    def collate_fn(self, batch):
        question = [i["question"] for i in batch]
        context = [i["context"] for i in batch]
        answer = [i["answer"] for i in batch]
        # doc_score=torch.stack([i["doc_score"] for i in batch],dim=0)
        return {
            "question": question,
            "context": context,
            "answer": answer,
            # "doc_score":doc_score,
        }


class NQOpenDatasetFactory:
    def __init__(
        self,
        data_dir="data/",
        trainset_file="trainset.pkl",
        testset_file="testset.pkl",
        device="cuda",
        train_json="NQ-open.train.jsonl",
        test_json="NQ-open.efficientqa.test.1.1.jsonl",
        train_context_json="train_context.jsonl",
        test_context_json="test_context.jsonl",
        trainset_size=20000,
        train_used=15000,
        testset_size=2000,
    ):
        self.device = device
        self.trainset_size = trainset_size
        self.train_used = train_used
        self.testset_size = testset_size
        self.trainset_path = os.path.join(data_dir, trainset_file)
        self.testset_path = os.path.join(data_dir, testset_file)
        self.train_json_path = os.path.join(data_dir, train_json)
        self.test_json_path = os.path.join(data_dir, test_json)
        self.train_context_json_path = os.path.join(
            data_dir, train_context_json)
        self.test_context_json_path = os.path.join(data_dir, test_context_json)

        # preprocessing
        if not os.path.exists(self.trainset_path):
            self.preprocess(os.path.join(data_dir, train_json), self.trainset_path,
                            self.train_context_json_path, self.trainset_size)
        if not os.path.exists(self.testset_path):
            self.preprocess(os.path.join(data_dir, test_json), self.testset_path,
                            self.test_context_json_path, self.testset_size)
        with open(self.trainset_path, "rb") as f:
            self.trainset = pickle.load(f)
            print(f"size of train+valid: {len(self.trainset["question"])}")
        with open(self.testset_path, "rb") as f:
            self.testset = pickle.load(f)
            print(f"size of test: {len(self.testset["question"])}")

    def gather_contexts(self, json_path, context_path, dataset_size):
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

    def preprocess(self, json_path, dump_path, context_path, dataset_size):
        encoders = ComprehensiveEncoder(self.device)
        data = {"question": [], "question_embedding": [],
                "context": [], "context_embedding": [], "answer": []}
        self.gather_contexts(json_path, context_path, dataset_size)
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
                        for k in range(0,len(words),50):
                            splitted_search_results.append(" ".join(words[k:min(k+50,len(words))]))
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

    def get_trainset(self,n_docs):
        return NQOpenDataset(self.trainset, 0, self.train_used, train=True,n_docs=n_docs)

    def get_valset(self,n_docs):
        return NQOpenDataset(self.trainset, self.train_used, -1, train=True,n_docs=n_docs)

    def get_testset(self,n_docs):
        return NQOpenDataset(self.testset, train=False,n_docs=n_docs)
