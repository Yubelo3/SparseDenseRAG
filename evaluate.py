import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from generator import RAGGenerator
from dataset import NQOpenDatasetFactory
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
import torch
import re
import bert_score

DEVICE = "cuda"
N_DOCS=2
CKPT=None
SETTING="original_2doc"

def preprocess(text):
    s = re.sub(r'[^\w\s]', ' ', text.lower())
    s = re.sub(r'\s+',' ',s).strip()
    tokens = word_tokenize(s)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def preprocess_without_stemming(text):
    s = re.sub(r'[^\w\s]', ' ', text.lower())
    s = re.sub(r'\s+',' ',s).strip()
    tokens = word_tokenize(s)
    return tokens

def main():
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    questions,answers,gt_answers=generate_answers()
    evaluation_result={
        "question":questions,
        "answer":answers,
        "gt_answers":gt_answers,
    }
    evaluation_result={"question":None,"answer":None,"gt":None,"exact_match":[],"bleu_score":[],"meteor_score":[],"bert_score":[]}
    evaluation_result["exact_match"]=metric_exact_match(answers,gt_answers)
    evaluation_result["bleu_score"]=metric_bleu_score(answers,gt_answers)
    evaluation_result["meteor_score"]=metric_meteor_score(answers,gt_answers)
    evaluation_result["bert_score"]=metric_bert_score(answers,gt_answers)
    with open(f"evaluation_result/{SETTING}.pkl","wb") as f:
        pickle.dump(evaluation_result,f)
    print(f"EXACT: {sum(evaluation_result['exact_match'])/len(evaluation_result['exact_match'])}")
    print(f"BLEU: {sum(evaluation_result['bleu_score'])/len(evaluation_result['bleu_score'])}")
    print(f"METEOR: {sum(evaluation_result['meteor_score'])/len(evaluation_result['meteor_score'])}")
    print(f"BERT: {sum(evaluation_result['bert_score'])/len(evaluation_result['bert_score'])}")
    

def metric_exact_match(answers,gt_answers):
    print("computing exact match ...")
    scores=[]
    for answer,gt_answer_list in tqdm(zip(answers,gt_answers)):
        clean_answer_str=" ".join(preprocess(answer))
        clean_gt=[preprocess(g) for g in gt_answer_list]
        score=0.0
        for g in clean_gt:
            clean_gt_str=" ".join(g)
            if clean_answer_str==clean_gt_str:
                score=1.0
                break
        scores.append(score)
    return scores

def metric_meteor_score(answers,gt_answers):
    print("computing bleu ...")
    scores=[]
    for answer,gt_answer_list in tqdm(zip(answers,gt_answers)):
        clean_answer=preprocess(answer)
        clean_gt=[preprocess(g) for g in gt_answer_list]
        score=meteor_score(clean_gt,clean_answer)
        scores.append(score)
    return scores


def metric_bleu_score(answers,gt_answers):
    print("computing meteor ...")
    scores=[]
    for answer,gt_answer_list in tqdm(zip(answers,gt_answers)):
        clean_answer=preprocess(answer)
        clean_gt=[preprocess(g) for g in gt_answer_list]
        score=sentence_bleu(clean_gt,clean_answer,weights=[1.0,0.0,0.0,0.0])
        scores.append(score)
    return scores


def metric_bert_score(answers,gt_answers):
    print("computing bert-score ...")
    clean_answers,clean_gts=[],[]
    for answer,gt_answer_list in tqdm(zip(answers,gt_answers)):
        clean_answers.append(" ".join(preprocess_without_stemming(answer)))
        clean_gts.append([" ".join(preprocess_without_stemming(g)) for g in gt_answer_list])
    P, R, F1 = bert_score.score(clean_answers,clean_gts, model_type="roberta-base",device=DEVICE)
    return F1


def generate_answers():
    generator = RAGGenerator(DEVICE)
    if CKPT is not None:
        generator.load_state_dict(torch.load(CKPT,map_location=DEVICE))
    generator.eval()
    dataset_factory = NQOpenDatasetFactory(device=DEVICE)
    testset=dataset_factory.get_testset(n_docs=N_DOCS)
    generator.model.config.n_docs=N_DOCS
    testloader=DataLoader(testset,batch_size=64,shuffle=False,collate_fn=testset.collate_fn)
    questions,answers,gt_answers=[],[],[]
    with torch.no_grad():
        for x in tqdm(testloader):
            questions+=x["question"]
            gt_answers+=x["answer"]
            contexts=x["context"]
            results=generator.rag_injected_generate(x["question"],contexts)
            results=generator.decode_answers(results)
            answers+=results
    return questions,answers,gt_answers

    

if __name__ == "__main__":
    # print(sentence_bleu([["amma kacy"],["amma becky"]],["amma best"],weights=(1.0,0.0,0.0,0.0)))
    main()


