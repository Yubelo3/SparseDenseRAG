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
    def __init__(self,dataset,train=True) -> None:
        super().__init__()
        self.dataset=dataset
        self.train=train
    
    def __len__(self):
        return len(self.dataset["question"])

    def __getitem__(self, index):
        ret= {
            "question":self.dataset["question"][index],
            "context":self.dataset["context"][index],
            "answer":self.dataset["answer"][index]
        }
        if self.train:
            ret["answer"]=ret["answer"][0]
        return ret
    
    

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
        testset_size=2000,
    ):
        self.device=device
        self.trainset_size=trainset_size
        self.testset_size=testset_size
        self.trainset_path=os.path.join(data_dir,trainset_file)
        self.testset_path=os.path.join(data_dir,testset_file)
        self.train_json_path=os.path.join(data_dir,train_json)
        self.test_json_path=os.path.join(data_dir,test_json)
        self.train_context_json_path=os.path.join(data_dir,train_context_json)
        self.test_context_json_path=os.path.join(data_dir,test_context_json)

        # preprocessing
        if not os.path.exists(self.trainset_path):
            self.preprocess(os.path.join(data_dir,train_json),self.trainset_path,self.train_context_json_path,self.trainset_size)
        if not os.path.exists(self.testset_path):
            self.preprocess(os.path.join(data_dir,test_json),self.testset_path,self.test_context_json_path,self.testset_size)
        with open(self.trainset_path,"rb") as f:
            self.trainset=pickle.load(f)
        with open(self.testset_path,"rb") as f:
            self.testset=pickle.load(f)

    def gather_contexts(self,json_path,context_path,dataset_size):
        next_idx=0
        if os.path.exists(context_path):
            with jsonlines.open(context_path,"r") as reader:
                for line in reader:
                    next_idx=line[0]+1
        sparse_retriever=SparseRetriever()
        with jsonlines.open(json_path,"r") as reader:
            with jsonlines.open(context_path,"a") as writer:
                for i,sample in enumerate(tqdm(reader)):
                    if i>=dataset_size:
                        break
                    if i<next_idx:
                        continue
                    search_results=sparse_retriever.query(sample["question"],3)
                    writer.write([i,search_results])

    
    def preprocess(self,json_path,dump_path,context_path,dataset_size):
        encoders=ComprehensiveEncoder(self.device)
        data={"question":[],"question_embedding":[],"context":[],"context_embedding":[],"answer":[]}
        self.gather_contexts(json_path,context_path,dataset_size)
        with jsonlines.open(json_path,"r") as reader:
            with jsonlines.open(context_path,"r") as context_reader:
                for i,(sample,context) in enumerate(tqdm(zip(reader,context_reader))):
                    assert i==context[0]
                    question=sample["question"]
                    question_embedding:torch.Tensor = encoders.encode_questions([question])
                    search_results=context[1]
                    if(len(search_results)<10):
                        print(f"WARNING: not sufficient context for {i}")
                        print(f"question is {sample["question"]}")
                        continue
                    context_embedding:torch.Tensor=encoders.encode_paragraphs(search_results)
                    dis = torch.norm(question_embedding-context_embedding, p=2, dim=-1)
                    _, relevant_context_indices = torch.topk(dis, k=10, largest=False, sorted=True)
                    data["question"].append(question)
                    data["question_embedding"].append(question_embedding.detach().cpu().numpy())
                    data["context"].append([search_results[i] for i in relevant_context_indices.cpu().tolist()])
                    data["context_embedding"].append(context_embedding[relevant_context_indices].cpu().numpy())
                    
                    if "answer_and_def_correct_predictions" in sample:
                        answers=sample["answer_and_def_correct_predictions"]
                    else:
                        answers=sample["answer"]
                    data["answer"].append(answers)
                    if (i+1)%1000==0:
                        partial_dump_path=f"{dump_path.split(".")[0]}-{i+1}.{dump_path.split(".")[1]}"
                        with open(partial_dump_path,"wb") as f:
                            pickle.dump(data,f)
        with open(dump_path,"wb") as f:
            pickle.dump(data,f)
    
    def get_trainset(self):
        return NQOpenDataset(self.trainset,train=True)
    
    def get_testset(self):
        return NQOpenDataset(self.testset,train=False)
            
