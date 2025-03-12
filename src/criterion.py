# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging.meters import safe_round


@register_criterion("decoder_only_language_modeling_loss", dataclass=FairseqDataclass)
class decoder_only_language_modeling_loss(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        loss, lprobs, llm_labels = model(target_list=sample["target"], **sample["net_input"])
        sample_size = (
            len(sample["target"])
        )

        n_correct, total = self.compute_accuracy(lprobs, llm_labels)
        
        logging_output = {
            "loss": loss.item(),
            "ntokens": total,
            "nsentences": len(sample["target"]),
            "sample_size": sample_size,
        }
        
        logging_output["n_correct"] = utils.item(n_correct.data)
        logging_output["total"] = utils.item(total.data)

        if not model.training:
            import editdistance
            n_err = 0
            n_total = 0
            with torch.no_grad():
                refs = model.tokenizer.batch_decode(sample['target'],
                                                   skip_special_tokens=True, 
                                                   clean_up_tokenization_spaces=False)
                best_hypo = model.generate(**sample["net_input"], num_beams=5, temperature=0.3)
                hypos = model.tokenizer.batch_decode(best_hypo, 
                                                     skip_special_tokens=True, 
                                                     clean_up_tokenization_spaces=False)
                
            for hypo, ref in zip(hypos, refs):
                hypo, ref = hypo.strip().split(), ref.strip().split()
                n_err += editdistance.eval(hypo, ref)
                n_total += len(ref)
            
            logging_output["n_err"] = n_err
            logging_output["n_total"] = n_total
            
        del lprobs, n_correct, total
        
        return loss, sample_size, logging_output

    
    def compute_accuracy(self, lprobs, labels):
        shifted_logits = lprobs[:, :-1, :]
        shifted_labels = labels[:, 1:]      
        
        predictions = torch.argmax(shifted_logits, dim=-1)
        mask = shifted_labels != -100
        
        correct_predictions = (predictions == shifted_labels) & mask
        
        n_correct = correct_predictions.sum().float()
        total = mask.sum().float()

        return n_correct, total

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

        n_err = sum(log.get("n_err", 0) for log in logging_outputs)
        metrics.log_scalar("_n_err", n_err)
        n_total = sum(log.get("n_total", 0) for log in logging_outputs)
        metrics.log_scalar("_n_total", n_total)

        if n_err > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_n_err"].sum * 100.0 / meters["_n_total"].sum, 3
                )
                if meters["_n_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        # """
        # Whether the logging outputs returned by `forward` can be summed
        # across workers prior to calling `reduce_metrics`. Setting this
        # to True will improves distributed training speed.
        # """
        return False

