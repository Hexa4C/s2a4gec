# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.float().gather(dim=-1, index=target)
    smooth_loss = -lprobs.float().sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    loss = loss.type_as(lprobs)
    nll_loss = nll_loss.type_as(lprobs)
    return loss, nll_loss


def s2a_nllloss(lprobs, labels, epsilon, reduce=True, norm=False):
    if labels.dim() == lprobs.dim() - 1:
        labels = labels.unsqueeze(-1)
    pad_mask = labels.eq(-1)
    if norm:
        assert reduce == True
        skip_mask = labels.ne(0)
        labels[pad_mask] = 0
        skip_loss = -lprobs.gather(dim=-1, index=labels)
        skip_loss.masked_fill_(skip_mask, 0.)
        copy_mask = labels.ne(1)
        copy_loss = -lprobs.gather(dim=-1, index=labels)
        copy_loss.masked_fill_(copy_mask, 0.)
        gen_mask = labels.ne(2)
        gen_loss = -lprobs.gather(dim=-1, index=labels)
        gen_loss.masked_fill_(gen_mask, 0.)

        skip_cnt = (skip_mask == False).sum()
        copy_cnt = (copy_mask == False).sum()
        gen_cnt = (gen_mask == False).sum()
        total_cnt = (pad_mask == False).sum()
        nll_loss = (skip_loss.sum() / skip_cnt + copy_loss.sum() / copy_cnt + gen_loss.sum() / gen_cnt) * total_cnt
        return nll_loss
    else:
        labels[pad_mask] = 0
        nll_loss = -lprobs.gather(dim=-1, index=labels)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = epsilon / lprobs.size(-1)
        loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss
        # return nll_loss


@register_criterion('s2a_loss')
class S2ALabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, s2a_label_smoothing, s2a_loss):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.s2a_eps = s2a_label_smoothing
        self.gamma = s2a_loss

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--s2a-label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for s2a label smoothing, 0 means no label smoothing')
        parser.add_argument('--s2a-loss', default=1., type=float, metavar='D',
                            help='hyperparameter for loss of sequence labeling')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])

        s2a_lprobs, lprobs = model.get_normalized_probs(net_output, log_probs=True, with_orig_lprobs=True)
        s2a_lprobs = s2a_lprobs.view(-1, s2a_lprobs.size(-1))
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)

        gen_loss, gen_nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        s2a_loss, s2a_nll_loss = label_smoothed_nll_loss(
            s2a_lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        # loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        # gen_loss = loss.data
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']

        s2a_labels = sample['labels']
        s2a_out = net_output[1]['s2a_out']
        label_lprobs = F.log_softmax(s2a_out, dim=-1)
        _, label_loss = s2a_nllloss(label_lprobs, s2a_labels, self.s2a_eps, reduce=reduce)
        # loss = gen_loss + label_loss
        # loss = s2a_loss + self.gamma * gen_loss
        loss = (1 - self.gamma) * s2a_loss + self.gamma * gen_loss
        # loss = s2a_loss + self.gamma * (gen_loss + label_loss)
        nll_loss = s2a_nll_loss

        logging_output = {
            'loss': loss.float().data,
            'nll_loss': nll_loss.float().data,
            'gen_loss': gen_loss.float().data,
            'gen_nll_loss': gen_nll_loss.float().data,
            's2a_loss': s2a_loss.float().data,
            's2a_nll_loss': s2a_nll_loss.float().data,
            'label_loss': label_loss.float().data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        gen_loss_sum = sum(log.get('gen_loss', 0) for log in logging_outputs)
        gen_nll_loss_sum = sum(log.get('gen_nll_loss', 0) for log in logging_outputs)
        s2a_loss_sum = sum(log.get('s2a_loss', 0) for log in logging_outputs)
        s2a_nll_loss_sum = sum(log.get('s2a_nll_loss', 0) for log in logging_outputs)
        label_loss_sum = sum(log.get('label_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=4)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('gen_loss', gen_loss_sum / ntokens / math.log(2), sample_size, round=3)
        metrics.log_scalar('gen_nll_loss', gen_nll_loss_sum / ntokens / math.log(2), sample_size, round=3)
        metrics.log_scalar('s2a_loss', s2a_loss_sum / ntokens / math.log(2), sample_size, round=3)
        metrics.log_scalar('s2a_nll_loss', s2a_nll_loss_sum / ntokens / math.log(2), sample_size, round=5)
        metrics.log_scalar('label_loss', label_loss_sum / ntokens / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg, round=4))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
