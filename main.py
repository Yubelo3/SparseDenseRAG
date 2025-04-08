
import torch
from sparse_retriever import SparseRetriever
from comprehensive_encoder import ComprehensiveEncoder

question = "who is the first programmer?"

sparse_net_retriever = SparseRetriever()
encoders = ComprehensiveEncoder()
knowledge = sparse_net_retriever.query(question)
knowledge_embedding = encoders.encode_paragraphs(knowledge)
question_embedding = encoders.encode_questions([question])
print(knowledge_embedding)

dis = torch.norm(question_embedding-knowledge_embedding, p=2, dim=-1)
_, nearest_knowledge = torch.topk(dis, k=10, largest=False, sorted=True)
nearest_knowledge = nearest_knowledge.tolist()
print("-"*50)
for i in nearest_knowledge:
    print(knowledge[i])
