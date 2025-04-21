import torch
from typing import List
from transformers import BartForConditionalGeneration, BartTokenizer, BartTokenizerFast,RagSequenceForGeneration, RagTokenizer, RagRetriever,BartForConditionalGeneration
from transformers import RagModel
from typing import Optional
from transformers.utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.bart.modeling_bart import BART_INPUTS_DOCSTRING,_CONFIG_FOR_DOC,BART_GENERATION_EXAMPLE,shift_tokens_right
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Union,Tuple



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
        answers:List[str],
        reduction:bool=False
    ):
        B=len(questions)
        encoder_str,decoder_str=[],[]
        for q,ctxs,a in zip(questions,contexts,answers):
            for c in ctxs:
                encoder_str.append(" "+self.model.config.title_sep+c+self.model.config.doc_sep+q)
                decoder_str.append(a)
        encoder_tensor,encoder_mask=self.tokenize_context(encoder_str)
        decoder_tensor,decoder_mask=self.tokenize_context(decoder_str)
        if reduction:
            return self.model.generator.forward(
                input_ids=encoder_tensor,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_tensor[:,:-1].contiguous(),
                decoder_attention_mask=decoder_mask[:,:-1].contiguous(),
                labels=decoder_tensor[:,1:].clone(),
                return_dict=True
            )["loss"]
        else:
            return self.loss_non_redution_forward(
                input_ids=encoder_tensor,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_tensor[:,:-1].contiguous(),
                decoder_attention_mask=decoder_mask[:,:-1].contiguous(),
                labels=decoder_tensor[:,1:].clone(),
                return_dict=True
            )["loss"].view(B,-1).mean(dim=-1)

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

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def loss_non_redution_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.model.generator.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.model.generator.config.pad_token_id, self.model.generator.config.decoder_start_token_id
                )

        outputs = self.model.generator.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.model.generator.lm_head(outputs[0])
        lm_logits = lm_logits + self.model.generator.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss(reduction="none")
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.model.generator.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


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