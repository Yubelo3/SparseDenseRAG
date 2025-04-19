import torch
from typing import Optional
from transformers import RagSequenceForGeneration

@torch.no_grad()
def generate(
    model:RagSequenceForGeneration,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    context_input_ids: Optional[torch.LongTensor] = None,
    context_attention_mask: Optional[torch.LongTensor] = None,
    doc_scores: Optional[torch.FloatTensor] = None,
    do_deduplication: Optional[bool] = None,  # defaults to True
    num_return_sequences: Optional[int] = None,  # defaults to 1
    num_beams: Optional[int] = None,  # defaults to 1
    n_docs: Optional[int] = None,
    **model_kwargs,
) -> torch.LongTensor:
    n_docs = n_docs if n_docs is not None else model.config.n_docs
    model.config.n_docs=1
    n_docs=1
    do_deduplication = do_deduplication if do_deduplication is not None else model.config.do_deduplication
    num_doc_return_sequences = (
        num_return_sequences if num_return_sequences is not None else model.config.num_return_sequences
    )
    num_beams = num_beams if num_beams is not None else model.config.num_beams
    hypos = []
    model_kwargs["num_beams"] = num_beams
    model_kwargs["num_return_sequences"] = num_beams
    model_kwargs["attention_mask"] = None
    batch_size = input_ids.shape[0] if input_ids is not None else context_input_ids.shape[0] // n_docs
    for index in range(batch_size):
        # first, generate beams from documents:
        generator_input_ids = context_input_ids[index * n_docs : (index + 1) * n_docs]  # (n_docs, max_len)
        generator_attention_mask= context_attention_mask[index * n_docs : (index + 1) * n_docs]
        output_sequences = model.generator.generate(
            generator_input_ids,
            **model_kwargs,
        )  # n_docs * n_beam, tgt_len
        # n_docs=5, n_beam=4, tgt_len=7?
        # this is good enough
        if do_deduplication:
            # do_deduplication, max_output_len
            output_sequences = torch.stack(list({str(k.tolist()): k for k in output_sequences}.values()))
        num_candidates = output_sequences.shape[
            0
        ]  # after deduplication, this number can be less than n_docs*n_beam
        repeated_context_input_ids=generator_input_ids.repeat(num_candidates,1)
        repeated_context_attention_mask=generator_attention_mask.repeat(num_candidates,1)
        # then, run model forwards to get nll scores:
        new_input_ids = input_ids[index : index + 1].repeat(num_candidates, 1)
        outputs = model(
            new_input_ids,
            context_input_ids=repeated_context_input_ids,
            context_attention_mask=repeated_context_attention_mask,
            labels=output_sequences,
            doc_scores=doc_scores,
            exclude_bos_score=True)
        top_cand_inds = (-outputs["loss"]).topk(num_doc_return_sequences)[1]

        hypos.append(output_sequences[top_cand_inds])

    return model._cat_and_pad(hypos, pad_token_id=model.config.generator.pad_token_id)

