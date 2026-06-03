import transformers
from packaging import version
import torch
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.generation.utils import (
    GenerationMixin,
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
)
import torch.nn as nn
import logging
from typing import Optional, Union, TYPE_CHECKING

from .beam_search import BeamSearchScorer

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = logging.getLogger(__name__)

class HammingDiversityLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces diverse beam search.

    Note that this logits processor is only effective for [`PreTrainedModel.group_beam_search`]. See [Diverse Beam
    Search: Decoding Diverse Solutions from Neural Sequence Models](https://huggingface.co/papers/1610.02424) for more
    details.

    Traditional beam search often generates very similar sequences across different beams.
    `HammingDiversityLogitsProcessor` addresses this by penalizing beams that generate tokens already chosen by other
    beams in the same time step.

    Args:
        diversity_penalty (`float`):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. A higher `diversity_penalty` will enforce greater diversity among the beams. Adjusting
            this value can help strike a balance between diversity and natural likelihood.
        num_beams (`int`):
            Number of beams for beam search. 1 means no beam search.
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            [this paper](https://huggingface.co/papers/1610.02424) for more details.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    >>> import torch

    >>> # Initialize the model and tokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

    >>> # A long text about the solar system
    >>> text = (
    ...     "The Solar System is a gravitationally bound system comprising the Sun and the objects that orbit it, "
    ...     "either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight "
    ...     "planets, with the remainder being smaller objects, such as the five dwarf planets and small Solar System "
    ...     "bodies. The Solar System formed 4.6 billion years ago from the gravitational collapse of a giant "
    ...     "interstellar molecular cloud."
    ... )
    >>> inputs = tokenizer("summarize: " + text, return_tensors="pt")

    >>> # Generate diverse summary
    >>> outputs_diverse = model.generate(
    ...     **inputs,
    ...     num_beam_groups=2,
    ...     diversity_penalty=10.0,
    ...     max_length=100,
    ...     num_beams=4,
    ...     num_return_sequences=2,
    ... )
    >>> summaries_diverse = tokenizer.batch_decode(outputs_diverse, skip_special_tokens=True)

    >>> # Generate non-diverse summary
    >>> outputs_non_diverse = model.generate(
    ...     **inputs,
    ...     max_length=100,
    ...     num_beams=4,
    ...     num_return_sequences=2,
    ... )
    >>> summary_non_diverse = tokenizer.batch_decode(outputs_non_diverse, skip_special_tokens=True)

    >>> # With `diversity_penalty`, the resulting beams are much more diverse
    >>> print(summary_non_diverse)
    ['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
    'the Solar System formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.']

    >>> print(summaries_diverse)
    ['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
    'the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets. the rest of the objects are smaller objects, such as the five dwarf planets and small solar system bodies.']
    ```
    """

    def __init__(self, diversity_penalty: float, num_beams: int, num_beam_groups: int):
        if not isinstance(diversity_penalty, float) or (not diversity_penalty > 0.0):
            raise ValueError("`diversity_penalty` should be a float strictly larger than 0.")
        self._diversity_penalty = diversity_penalty
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        if num_beam_groups > num_beams:
            raise ValueError("`beam_groups` has to be smaller or equal to `num_beams`.")
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        current_tokens: torch.LongTensor,
        beam_group_idx: int,
    ) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            current_tokens (`torch.LongTensor` of shape `(batch_size)`):
                Indices of input sequence tokens in the vocabulary, corresponding to the tokens selected by the other
                beam groups in the current generation step.
            beam_group_idx (`int`):
                The index of the beam group currently being processed.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.
        """
        # hamming diversity: penalise using same token in current group which was used in previous groups at
        # the same time step
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]

        if group_start_idx == 0:
            return scores

        scores_processed = scores.clone()
        for batch_idx in range(batch_size):
            # predicted tokens of last time step of previous groups
            previous_group_tokens = current_tokens[
                batch_idx * self._num_beams : batch_idx * self._num_beams + group_start_idx
            ]
            token_frequency = torch.bincount(previous_group_tokens, minlength=vocab_size).to(scores.device)
            scores_processed[batch_idx * group_size : (batch_idx + 1) * group_size] -= (
                self._diversity_penalty * token_frequency
            )

        return scores_processed


def _group_beam_search(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
):
    r"""
    Generates sequences of token ids for models with a language modeling head using **diverse beam search
    decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size*num_beams, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        model_kwargs:
            Additional model specific kwargs that will be forwarded to the `forward` function of the model. If
            model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # check parameters
    assert (
        generation_config.diversity_penalty != 0.0 and generation_config.num_beam_groups != 1
    ), "Group beam search requires diversity_penalty > 0.0 and num_beam_groups > 1"
    if generation_config.do_sample is True:
        raise ValueError("Group beam search requires `do_sample` to be set to `False`")
    if generation_config.num_beams % generation_config.num_beam_groups != 0:
        raise ValueError("Group beam search requires `num_beams` to be divisible by `num_beam_groups`")
    if generation_config.diversity_penalty == 0.0:
        raise ValueError("Group beam search requires `diversity_penalty` to be greater than `0.0`, otherwise your groups will be identical.")

    if streamer is not None:
        raise ValueError("Group beam search does not support streaming")

    if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
        logits_processor.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=generation_config.diversity_penalty,
                num_beams=generation_config.num_beams,
                num_beam_groups=generation_config.num_beam_groups,
            )
        )

    # define beam scorer
    beam_scorer = BeamSearchScorer(
        batch_size=input_ids.shape[0] // generation_config.num_beams,
        num_beams=generation_config.num_beams,
        device=input_ids.device,
        length_penalty=generation_config.length_penalty,
        do_early_stopping=generation_config.early_stopping,
        num_beam_hyps_to_keep=generation_config.num_return_sequences,
        num_beam_groups=generation_config.num_beam_groups,
        max_length=generation_config.max_length,
    )
    # init values
    pad_token_id = generation_config._pad_token_tensor
    eos_token_id = generation_config._eos_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate

    num_beams = beam_scorer.num_beams
    num_beam_groups = beam_scorer.num_beam_groups
    num_sub_beams = num_beams // num_beam_groups
    batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
    device = input_ids.device

    batch_beam_size, cur_len = input_ids.shape
    # Does not exist anymore in recent versions!
    if hasattr(model, "_get_initial_cache_position"):
        model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    if return_dict_in_generate and output_scores:
        beam_indices = [
            tuple(() for _ in range(num_sub_beams * batch_size))
            for _ in range(num_beam_groups)
        ]
    else:
        beam_indices = None

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # initialise score of first beam of each group with 0 and the rest with -1e9. This ensures that the beams in
    # the same group don't produce same tokens every time.
    beam_scores = torch.full(
        (batch_size, num_beams), -1e9, dtype=torch.float, device=device
    )
    beam_scores[:, ::num_sub_beams] = 0
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False

    decoder_prompt_len = input_ids.shape[1]  # record the prompt length of decoder
    is_transformers_geq_5_3_0 = version.parse(transformers.__version__) >= version.parse("5.3.0")

    prefill_consumed = False
    if is_transformers_geq_5_3_0:
        outputs = model._prefill(
            input_ids,
            generation_config,
            model_kwargs,
            is_first_iteration=not getattr(generation_config, "is_assistant", False),
        )
    else:
        outputs = None

    model_forward = model.__call__
    if hasattr(model, "get_compiled_call") and hasattr(model, "_valid_auto_compile_criteria"):
        if model._valid_auto_compile_criteria(model_kwargs, generation_config):
            model_forward = model.get_compiled_call(generation_config.compile_config)

    while model._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        # predicted tokens in cur_len step
        current_tokens = torch.zeros(
            batch_size * num_beams, dtype=input_ids.dtype, device=device
        )

        # indices which will form the beams in the next time step
        reordering_indices = torch.zeros(
            batch_size * num_beams, dtype=torch.long, device=device
        )

        if not is_transformers_geq_5_3_0 or prefill_consumed:
            # do one decoder step on all beams of all sentences in batch
            if is_transformers_geq_5_3_0:
                next_sequence_length = 1 if model_kwargs.get("use_cache", False) else None
                model_inputs = model.prepare_inputs_for_generation(
                    input_ids, next_sequence_length=next_sequence_length, **model_kwargs
                )
            else:
                model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs.update(
                {"output_hidden_states": output_hidden_states}
                if output_hidden_states
                else {}
            )

            if hasattr(model, "_optimize_model_for_decode"):
                with model._optimize_model_for_decode():
                    outputs = model_forward(**model_inputs, return_dict=True)
            else:
                outputs = model_forward(**model_inputs, return_dict=True)
        prefill_consumed = True

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue

        if output_scores:
            processed_score = torch.zeros_like(outputs.logits[:, -1, :])
        if output_logits:
            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            raw_logit_score = outputs.logits[:, -1, :].to(
                copy=True, device=input_ids.device
            )

        for beam_group_idx in range(num_beam_groups):
            group_start_idx = beam_group_idx * num_sub_beams
            group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
            group_size = group_end_idx - group_start_idx

            # indices of beams of current group among all sentences in batch
            batch_group_indices = []

            for batch_idx in range(batch_size):
                batch_group_indices.extend(
                    [
                        batch_idx * num_beams + idx
                        for idx in range(group_start_idx, group_end_idx)
                    ]
                )
            group_input_ids = input_ids[batch_group_indices]

            # select outputs of beams of current group only
            # copy=True is needed to avoid keeping a hanging reference to outputs.logits which may be very large for the first iteration
            # .float() is needed to retain precision for later logits manipulations
            next_token_logits = outputs.logits[batch_group_indices, -1, :].to(
                copy=True, dtype=torch.float32, device=input_ids.device
            )

            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * group_size, vocab_size)
            vocab_size = next_token_scores.shape[-1]

            next_token_scores_processed = logits_processor(
                group_input_ids,
                next_token_scores,
                current_tokens=current_tokens,
                beam_group_idx=beam_group_idx,
            )
            next_token_scores = next_token_scores_processed + beam_scores[
                batch_group_indices
            ].unsqueeze(-1)
            next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

            if output_scores:
                processed_score[batch_group_indices] = next_token_scores_processed.to(processed_score.dtype)


            # reshape for beam search
            next_token_scores = next_token_scores.view(
                batch_size, group_size * vocab_size
            )

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores,
                max(2, 1 + n_eos_tokens) * group_size,
                dim=1,
                largest=True,
                sorted=True,
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            process_beam_indices = (
                sum(beam_indices, ()) if beam_indices is not None else None
            )
            beam_outputs = beam_scorer.process(
                group_input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=process_beam_indices,
                group_index=beam_group_idx,
                decoder_prompt_len=decoder_prompt_len,
            )
            beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            if return_dict_in_generate and output_scores:
                beam_indices[beam_group_idx] = tuple(
                    beam_indices[beam_group_idx][beam_idx[i]] + (beam_idx[i],)
                    for i in range(len(beam_indices[0]))
                )

            input_ids[batch_group_indices] = group_input_ids[beam_idx]
            group_input_ids = torch.cat(
                [group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )
            current_tokens[batch_group_indices] = group_input_ids[:, -1]

            # (beam_idx // group_size) -> batch_idx
            # (beam_idx % group_size) -> offset of idx inside the group
            reordering_indices[batch_group_indices] = (
                num_beams * torch.div(beam_idx, group_size, rounding_mode="floor")
                + group_start_idx
                + (beam_idx % group_size)
            )

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (processed_score,)
            if output_logits:
                raw_logits += (raw_logit_score,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if model.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
        # (that way the memory peak does not include outputs.logits)
        del outputs

        # NOTE: we need to check if `model._reorder_cache` exists for special models like RAG, RecurrentGemma etc.
        if model_kwargs.get("past_key_values", None) is not None:
            if hasattr(model, "_reorder_cache"):
                model_kwargs["past_key_values"] = model._reorder_cache(
                    model_kwargs["past_key_values"], reordering_indices
                )
            else:
                model_kwargs["past_key_values"].reorder_cache(reordering_indices)

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
            this_peer_finished = True

    final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=final_beam_indices,
        decoder_prompt_len=decoder_prompt_len,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if model.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateBeamDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return sequence_outputs["sequences"]


def generate(model, *args, **kwargs):
    """Custom generate function for group beam search decoding.
    Args:
        model (`PreTrainedModel`):
            The model to generate from.
        num_beams (`int`): The number of beams to use for beam search.
        num_beam_groups (`int`): The number of beam groups to use for beam search.
        length_penalty (`float`): The length penalty to use for beam search.
        early_stopping (`bool`): Whether to stop beam search when sufficient beams have finished.
        num_return_sequences (`int`): The number of sequences to return.
        max_length (`int`): The maximum length of the generated sequence.
    """
    generation_outputs = GenerationMixin.generate(
        model, *args, custom_generate=_group_beam_search, **kwargs
    )
    return generation_outputs
