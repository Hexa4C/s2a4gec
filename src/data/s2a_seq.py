# author: Jiquan Li
# email: lijiquan@mail.ustc.edu.cn

import torch
from torch import Tensor
from .seq2labels import align_sequences


SKIP, CONN, GEN = 0, 1, 2


def combine_tokens(src_tokens, alignments):
    comb_tokens = []
    mask = []
    for src, actions in zip(src_tokens, alignments[1:]):
        if actions[0] == "$KEEP":
            comb_tokens.append(src)
            mask.append(CONN)
        elif actions[0] == "$DELETE":
            comb_tokens.append(src)
            mask.append(SKIP)
        elif actions[0][:8] == "$REPLACE":
            for a in actions:
                comb_tokens.append(a.split("_")[1])
                mask.append(GEN)
            comb_tokens.append(src)
            mask.append(SKIP)
        else:
            comb_tokens.append(src)
            mask.append(CONN)
            for a in actions:
                comb_tokens.append(a.split("_")[1])
                mask.append(GEN)
    return comb_tokens, mask


def get_succ_tokens(comb_tokens, mask):
    last_token = None
    succ_tokens = []
    for tok, m in zip(reversed(comb_tokens), reversed(mask)):
        if m == GEN:
            assert last_token is not None
            succ_tokens.append(last_token)
        else:
            last_token = tok
            succ_tokens.append(tok)
    return list(reversed(succ_tokens))


def get_prec_tokens(comb_tokens, mask):
    last_token = None
    prec_tokens = []
    for tok, m in zip(comb_tokens, mask):
        if m == SKIP:
            assert last_token is not None
            prec_tokens.append(last_token)
        else:
            last_token = tok
            prec_tokens.append(tok)
    return prec_tokens


def s2a_gen(src_tokens: Tensor, tgt_tokens: Tensor, blk_idx=-1, input_blk=False):
    src_list = [str(i) for i in src_tokens.tolist()]
    tgt_list = [str(i) for i in tgt_tokens.tolist()]
    alignments = align_sequences(" ".join(src_list), " ".join(tgt_list))    # get aligned edit operations
    comb_tokens, comb_mask = combine_tokens(src_list, alignments)       # combining x and y into one sequence
    comb_tokens = [int(i) for i in comb_tokens]
    succ_tokens = get_succ_tokens(comb_tokens, comb_mask)           # tilde x
    prec_tokens = get_prec_tokens(comb_tokens, comb_mask)
    actions = ["SKIP", "CONN", "GEN"]
    prec_out = torch.Tensor(prec_tokens[:-1]).to(src_tokens)           # tilde y_in
    succ_out = torch.Tensor(succ_tokens[1:]).to(src_tokens)
    mask_out = torch.Tensor(comb_mask[1:]).to(src_tokens)
    target_out = torch.Tensor(comb_tokens[1:]).to(src_tokens)       # tilde y_out
    target_out[mask_out.eq(SKIP)] = blk_idx
    if input_blk:
        prec_out = torch.cat((prec_out[0:1], target_out[:-1]))
    return prec_out, succ_out, mask_out, target_out


def test_main():
    src = torch.Tensor([2, 11, 13, 15, 12, 14, 15, 19, 20, 3]).long()
    tgt = torch.Tensor([2, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 3]).long()
    res = s2a_gen(src, tgt, input_blk=True)
    print(res)


if __name__ == "__main__":
    test_main()