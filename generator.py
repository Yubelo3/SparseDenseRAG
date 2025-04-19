import torch
from typing import List
from transformers import BartForConditionalGeneration, BartTokenizer, BartTokenizerFast,RagSequenceForGeneration, RagTokenizer, RagRetriever,BartForConditionalGeneration
from transformers import RagModel
from typing import Optional



class RAGGenerator:
    def __init__(self, device="cuda", model="facebook/rag-sequence-nq", ckpt=None,n_docs=1):
        self.device = device
        self.tokenizer = RagTokenizer.from_pretrained(model)
        self.model: RagSequenceForGeneration = RagSequenceForGeneration.from_pretrained(model, retriever=None).to(device)
        if ckpt != None:
            self.model.load_state_dict(torch.load(ckpt, map_location=device))
        for param in self.model.generator.model.encoder.layers.parameters():
            param.requires_grad = False
        for param in self.model.generator.model.encoder.layers[-1:].parameters():
            param.requires_grad = True
        for param in self.model.generator.model.decoder.parameters():
            param.requires_grad = False
        for param in self.model.generator.model.decoder.layers[-1:].parameters():
            param.requires_grad = True


    def load_state_dict(self,state_dict):
        self.model.load_state_dict(state_dict)

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def tokenize_question(self,content:List[str]):
        tokenized= self.tokenizer.prepare_seq2seq_batch(content,return_tensors="pt",padding=True)
        return tokenized["input_ids"].to(self.device),tokenized["attention_mask"].to(self.device)

    def tokenize_context(self,content:List[str]):
        tokenized=self.tokenizer.generator(content, return_tensors="pt",padding=True)
        return tokenized["input_ids"].to(self.device),tokenized["attention_mask"].to(self.device)
    

    # def tokenize_contexts(self,contexts:List[List[str]]):
    #     all_contexts=[]
    #     for c in contexts:
    #         all_contexts+=c
    #     ctx_dict=self.ctx_tokenizer(all_contexts,return_tensors="pt",padding=True,truncation=True,max_length=300)
    #     return ctx_dict["input_ids"].to(self.device),ctx_dict["attention_mask"].to(self.device)
    
    # def tokenize_questions(self,questions:List[str]):
    #     input_dict = self.tokenizer.prepare_seq2seq_batch(questions,return_tensors="pt",padding=True)
    #     return input_dict["input_ids"].to(self.device),input_dict["attention_mask"].to(self.device)

    # def tokenize_answers(self,answers:List[str]):
    #     output_dict=self.tokenizer.question_encoder.encode(answers,return_tensors="pt",padding=True)
    #     return output_dict["input_ids"].to(self.device),output_dict["attention_mask"].to(self.device)

    def get_loss(
        self,
        questions:List[str],
        contexts:List[List[str]],
        answers:List[str]
    ):
        encoder_str,decoder_str=[],[]
        for q,ctxs,a in zip(questions,contexts,answers):
            for c in ctxs:
                encoder_str.append(" "+self.model.config.title_sep+c+self.model.config.doc_sep+q)
                decoder_str.append(a)
        encoder_tensor,encoder_mask=self.tokenize_context(encoder_str)
        decoder_tensor,decoder_mask=self.tokenize_context(decoder_str)
        return self.model.generator.forward(
            input_ids=encoder_tensor,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_tensor[:,:-1].contiguous(),
            decoder_attention_mask=decoder_mask[:,:-1].contiguous(),
            labels=decoder_tensor[:,1:].clone(),
            return_dict=True
        )["loss"]

        # input_ids: Optional[torch.LongTensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # decoder_input_ids: Optional[torch.LongTensor] = None,
        # decoder_attention_mask: Optional[torch.LongTensor] = None,
        # head_mask: Optional[torch.Tensor] = None,
        # decoder_head_mask: Optional[torch.Tensor] = None,
        # cross_attn_head_mask: Optional[torch.Tensor] = None,
        # encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        # decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        # use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,



    @torch.no_grad()
    def rag_injected_generate(
        self,
        questions:List[str],
        contexts:List[List[str]],
        # doc_scores:torch.Tensor,
        **model_kwargs,
    ) -> torch.LongTensor:
        n_docs=len(contexts[0])
        # doc_scores=doc_scores.to(self.device)
        self.model.config.n_docs=n_docs
        # print(n_docs)
        num_beams = self.model.config.num_beams
        hypos = []
        model_kwargs["num_beams"] = num_beams
        model_kwargs["num_return_sequences"] = num_beams
        model_kwargs["attention_mask"] = None
        batch_size = len(questions)

        # tokenize
        input_ids,attention_mask=self.tokenize_question(questions)
        all_context=[]
        for q,c in zip(questions,contexts):
            for cc in c:
                all_context.append(" "+self.model.config.title_sep+cc+self.model.config.doc_sep+q)
        context_input_ids,context_attention_mask=self.tokenize_context(all_context)

        for index in range(batch_size):
            # first, generate beams from documents:
            generator_input_ids = context_input_ids[index * n_docs : (index + 1) * n_docs]  # (n_docs, max_len)
            generator_attention_mask= context_attention_mask[index * n_docs : (index + 1) * n_docs]
            output_sequences = self.model.generator.generate(
                generator_input_ids,
                **model_kwargs,
            )  # n_docs * n_beam, tgt_len
            # n_docs=5, n_beam=4, tgt_len=7?
            # this is good enough
            # deduplicated_sequences=set()
            # deduplicated_index=[]
            # for i,k in enumerate(output_sequences):
            #     k_str=str(k.tolist())
            #     if k_str not in deduplicated_sequences:
            #         deduplicated_sequences.add(k_str)
            #         deduplicated_index.append(i)
            # output_sequences=torch.stack(list(deduplicated_sequences))
            # deduplicated_index=torch.LongTensor(deduplicated_index)
            # context_score=doc_scores[index]
            output_sequences = torch.stack(list(
                {str(k.tolist()): k for k in output_sequences}.values()
            ))
            num_candidates = output_sequences.shape[0]
            repeated_context_input_ids=generator_input_ids.repeat(num_candidates,1)
            repeated_context_attention_mask=generator_attention_mask.repeat(num_candidates,1)
            # print(repeated_context_input_ids.shape)  # too large
            doc_scores = torch.ones((num_candidates,n_docs),device=self.device)*0.5
            # then, run model forwards to get nll scores:
            new_input_ids = input_ids[index : index + 1].repeat(num_candidates, 1)
            outputs = self.model(
                new_input_ids,
                context_input_ids=repeated_context_input_ids,
                context_attention_mask=repeated_context_attention_mask,
                labels=output_sequences,
                doc_scores=doc_scores,
                exclude_bos_score=True)
            top_cand_inds = (-outputs["loss"]).topk(1)[1]

            hypos.append(output_sequences[top_cand_inds])

        return self.model._cat_and_pad(hypos, pad_token_id=self.model.config.generator.pad_token_id)

    def decode_answers(self,answers):
        return self.tokenizer.batch_decode(answers,skip_special_tokens=True)



if __name__ == "__main__":
    generator = RAGGenerator(device="cpu")


# return RetrievAugLMMarginOutput(
#             loss=loss,
#             logits=outputs.logits,
#             doc_scores=outputs.doc_scores,
#             past_key_values=outputs.past_key_values,
#             context_input_ids=outputs.context_input_ids,
#             context_attention_mask=outputs.context_attention_mask,
#             retrieved_doc_embeds=outputs.retrieved_doc_embeds,
#             retrieved_doc_ids=outputs.retrieved_doc_ids,
#             question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
#             question_enc_hidden_states=outputs.question_enc_hidden_states,
#             question_enc_attentions=outputs.question_enc_attentions,
#             generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
#             generator_enc_hidden_states=outputs.generator_enc_hidden_states,
#             generator_enc_attentions=outputs.generator_enc_attentions,
#             generator_dec_hidden_states=outputs.generator_dec_hidden_states,
#             generator_dec_attentions=outputs.generator_dec_attentions,
#             generator_cross_attentions=outputs.generator_cross_attentions,
#         )