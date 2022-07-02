# This code is used for dynamically mask source input for gec
# author: Jiquan Li
# email: lijiquan@mail.ustc.edu.cn

from typing import Any, Dict, List
import torch
from torch import Tensor
import random
import numpy as np
import logging
from .dictionary import MaskedDictionary as CostomDictionary
from fairseq.utils import new_arange
import dimsim
import time
from pypinyin import lazy_pinyin
import ipdb

logger = logging.getLogger(__name__)

DICTIONARY_TYPE = CostomDictionary


class MaskScheme(object):
    def __init__(self, delta=0.3, p=0.2, span_max_len=5, span_min_len=1) -> None:
        super().__init__()
        self.delta = delta
        self.p = p
        self.lower = span_min_len
        self.upper = span_max_len
        self.lens = list(range(self.lower, self.upper + 1))
        self.len_distrib = [self.p * (1 - self.p) ** (i - self.lower) for i in
                            range(self.lower, self.upper + 1)] if self.p >= 0 else None
        self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib]

    @staticmethod
    def special_mask(src_tokens, dictionary):
        padding_mask = src_tokens.ne(dictionary.pad())  # mask origanl paddings in a batch
        bos_mask = src_tokens.ne(dictionary.bos())  # mask origanl bos in a batch
        eos_mask = src_tokens.ne(dictionary.eos())  # mask origanl eos in a batch
        sp_mask = padding_mask * bos_mask * eos_mask
        if hasattr(dictionary, 'cls'):
            cls_mask = src_tokens.ne(dictionary.cls())
            sp_mask = sp_mask * cls_mask
        if hasattr(dictionary, 'sep'):
            sep_mask = src_tokens.ne(dictionary.sep())
            sp_mask = sp_mask * sep_mask
        return sp_mask

    def mask(self, src_tokens: Tensor, dictionary: DICTIONARY_TYPE):
        sp_mask = self.special_mask(src_tokens, dictionary)
        sample_dist = torch.bernoulli(self.delta + torch.zeros_like(src_tokens)).bool()  # generate a 0/ 1 sample
        final_sample = sp_mask * sample_dist
        return final_sample

    def span_mask(self, src_tokens: Tensor, dictionary: DICTIONARY_TYPE):
        sp_mask = self.special_mask(src_tokens, dictionary)
        n_src = src_tokens.view(-1).size()[0]
        max_mask = int(n_src * self.delta)
        sample_dist = torch.zeros_like(src_tokens)
        n_masked = 0
        while n_masked < max_mask:
            span_len = np.random.choice(self.lens, p=self.len_distrib)
            left = np.random.choice(n_src)
            right = left + span_len
            for i in range(left, right):
                if n_masked > max_mask or i >= n_src:
                    break
                sample_dist[i] = 1
                n_masked += 1
        sample_dist = sample_dist.bool()
        final_sample = sp_mask * sample_dist
        return final_sample


class BaseAugmentor(object):
    def __init__(self, text_dict: DICTIONARY_TYPE) -> None:
        super().__init__()
        self.text_dict = text_dict

    def augment(self, src_tokens: Tensor, sample_dist: Tensor, **unused):
        raise NotImplementedError


class MaskAugmentor(BaseAugmentor):
    def __init__(self, text_dict: DICTIONARY_TYPE) -> None:
        super().__init__(text_dict)
        self.mask = text_dict.mask()

    def augment(self, src_tokens: Tensor, sample_dist: Tensor, **unused):
        return self.mask * sample_dist.int() + src_tokens * (1 - sample_dist.int())


class RandomAugmentor(BaseAugmentor):
    def __init__(self, text_dict: DICTIONARY_TYPE, special_token_num=5) -> None:
        super().__init__(text_dict)
        self.special_token_num = special_token_num
        self.vocab_size = len(self.text_dict) - self.special_token_num

    def augment(self, src_tokens: Tensor, sample_dist: Tensor, **unused):
        rand_vocabs = torch.randint_like(src_tokens, self.vocab_size) + self.special_token_num
        return rand_vocabs * sample_dist.int() + src_tokens * (1 - sample_dist.int())


class WordFreqAugmentor(BaseAugmentor):
    def __init__(self, text_dict: DICTIONARY_TYPE) -> None:
        super().__init__(text_dict)
        self.word_freq = torch.Tensor(text_dict.count)

    def augment(self, src_tokens: Tensor, sample_dist: Tensor, **unused):
        """
        sample substitions based on word frequency
        """
        freq_based_vocabs = torch.multinomial(torch.Tensor(self.word_freq), len(src_tokens.view(-1)), replacement=True)
        return freq_based_vocabs * sample_dist.int() + src_tokens * (1 - sample_dist.int())


class PinyinAugmentor(BaseAugmentor):
    def __init__(self, text_dict: DICTIONARY_TYPE, fuzzy=False, fuzzy_thre=2) -> None:
        super().__init__(text_dict)
        self.token2tokidx = {}  # token to token_idx
        self.pinyin2pyidx = {}  # pinyin to pinyin idx
        self.pinyin_list = []  # list of pinyin

        self.tokidx2pinyin = {}  # token idx to pinyin idx
        self.py_tok_set = {}  # key: pinyin idx, value: ([token idx], [freq])

        self.freqlist = []  # token idx to freq
        self.vocab_count = 0
        self.pinyin_count = 1

        self.pinyin2pyidx["1"] = 0
        self.py_tok_set["1"] = ([], [])
        self.pinyin_list.append("1")

        self.special_tokens = ["<pad>", "</s>", "<unk>", "<s>", "<mask>"]
        # for t in self.special_tokens:
        #     self._add_symbol(t, 0)

        self._load_pinyin_map()
        self.fuzzy = fuzzy
        if self.fuzzy:
            self._cal_pinyin_distance(fuzzy_thre)

    def _add_symbol(self, token, freq):
        if token in self.token2tokidx.keys():
            return
        pinyin = lazy_pinyin(token, errors='ignore')
        real_pinyin = "".join(pinyin) + "1"
        self.token2tokidx[token] = self.vocab_count
        tok_idx = self.vocab_count
        self.vocab_count += 1
        self.freqlist.append(freq)
        if real_pinyin == "hm1":
            real_pinyin = "hen1"

        if real_pinyin not in self.pinyin_list:
            self.pinyin2pyidx[real_pinyin] = self.pinyin_count
            py_idx = self.pinyin_count
            self.pinyin_list.append(real_pinyin)
            self.pinyin_count += 1
        else:
            py_idx = self.pinyin2pyidx[real_pinyin]

        self.tokidx2pinyin[tok_idx] = py_idx
        if py_idx not in self.py_tok_set.keys():
            self.py_tok_set[py_idx] = ([], [])
        self.py_tok_set[py_idx][0].append(tok_idx)
        self.py_tok_set[py_idx][1].append(freq)

    def _load_pinyin_map(self):
        for token, freq in zip(self.text_dict.symbols, self.text_dict.count):
            self._add_symbol(token, int(freq))

    def _cal_pinyin_distance(self, thre=5):
        self.fuzzy_pinyin = [[]]  # do not do fuzzy match for ""
        for i in range(1, self.pinyin_count):
            py_a = self.pinyin_list[i]
            self.fuzzy_pinyin.append([])
            for j in range(1, self.pinyin_count):
                py_b = self.pinyin_list[j]
                # try:
                if dimsim.get_distance([py_a], [py_b], pinyin=True) < thre:
                    self.fuzzy_pinyin[i].append(j)
                # except:
                #     print(py_a, py_b)
                #     quit()

    def augment(self, src_tokens: Tensor, sample_dist: Tensor, **unused):
        if self.fuzzy:
            return self.fuzzy_pinyin_substitution(src_tokens, sample_dist)
        else:
            return self.pinyin_substitution(src_tokens, sample_dist)

    def pinyin_substitution(self, src_tokens: Tensor, sample_dist: Tensor):
        result = []
        for i, atoken in enumerate(src_tokens.view(-1)):
            tok_idx = int(atoken)
            set_id = self.tokidx2pinyin[int(atoken)]
            if set_id == 0:
                result.append(int(atoken))
                continue
            asub = random.choices(self.py_tok_set[set_id][0], weights=self.py_tok_set[set_id][1], k=1)
            result.append(asub[0])
        final_subs = torch.Tensor(result).to(src_tokens)
        noised_tensor = final_subs * sample_dist.int() + src_tokens * (1 - sample_dist.int())
        return noised_tensor

    def fuzzy_pinyin_substitution(self, src_tokens: Tensor, sample_dist: Tensor):
        result = []
        for i, atoken in enumerate(src_tokens.view(-1)):
            set_id = self.tokidx2pinyin[int(atoken)]
            if set_id == 0:
                result.append(int(atoken))
                continue
            fuzzy_set_ids = self.fuzzy_pinyin[set_id]
            tok_set = []
            freq_set = []
            for id in fuzzy_set_ids:
                tok_set.extend(self.py_tok_set[id][0])
                freq_set.extend(self.py_tok_set[id][1])
            asub = random.choices(tok_set, weights=freq_set, k=1)
            result.append(asub[0])
        final_subs = torch.Tensor(result).to(src_tokens)
        return final_subs * sample_dist.int() + src_tokens * (1 - sample_dist.int())


class RandDelAugmentor(BaseAugmentor):
    def __init__(self, text_dict: DICTIONARY_TYPE, delta=0.0) -> None:
        super().__init__(text_dict)
        self.delta = delta

    def augment(self, src_tokens: Tensor, sample_dist: Tensor, **unused):
        padding_mask = src_tokens.eq(self.text_dict.pad())  # mask origanl paddings in a batch
        bos_mask = src_tokens.eq(self.text_dict.bos())  # mask origanl bos in a batch
        eos_mask = src_tokens.eq(self.text_dict.eos())  # mask origanl eos in a batch
        sample_dist_for_keep = (1 - torch.bernoulli(self.delta + torch.zeros_like(src_tokens))).bool()
        sample_dist_for_keep = sample_dist_for_keep | padding_mask | bos_mask | eos_mask
        return src_tokens[sample_dist_for_keep.nonzero()].view(-1)


class RandInsAugmentor(BaseAugmentor):
    def __init__(self, text_dict: DICTIONARY_TYPE, delta=0.0) -> None:
        super().__init__(text_dict)
        self.delta = delta

    def augment(self, src_tokens: Tensor, sample_dist: Tensor, **unused):
        """
        Only support one sample per call
        """
        # random insert <MASK> as placeholders
        padding_mask = src_tokens.ne(self.text_dict.pad())  # mask origanl paddings in a batch
        eos_mask = src_tokens.ne(self.text_dict.eos())  # mask origanl eos in a batch
        total_mask = padding_mask * eos_mask
        result = []
        for i, (atoken, mask) in enumerate(zip(src_tokens.view(-1), total_mask.view(-1))):
            result.append(atoken)
            ins_or_not = torch.rand((1,))
            if ins_or_not < self.delta and bool(mask):
                result.append(self.text_dict.mask())
        ins_tokens = torch.Tensor(result).to(src_tokens)
        # random select words to replace <MASK>
        placeholders = ins_tokens.eq(self.text_dict.mask())
        rand_vocabs = torch.randint_like(ins_tokens, len(self.text_dict) - 5) + 5
        return rand_vocabs * placeholders.int() + ins_tokens * (1 - placeholders.int())


class ShapeAugmentor(BaseAugmentor):
    def __init__(
            self, text_dict: DICTIONARY_TYPE,
            shape_file: str
    ) -> None:
        super().__init__(text_dict)
        self.word2idx, self.idx2word = {}, {}
        self.word_list, self.freq_list = [], []
        self.idx2shapeset = {}
        self.idx2shape_freq = {}
        self._build_shape_set(shape_file)

    def _build_shape_set(self, shape_file):
        self.word_list = self.text_dict.symbols
        self.freq_list = self.text_dict.count
        cnt = 0
        for aword in self.word_list:
            self.word2idx[aword] = cnt
            self.idx2word[cnt] = aword
            cnt += 1
        for word, idx in self.word2idx.items():
            self.idx2shapeset[idx] = []
            self.idx2shape_freq[idx] = []
            self.idx2shapeset[idx].append(idx)
            self.idx2shape_freq[idx].append(self.freq_list[idx])

        with open(shape_file, encoding='utf8') as f:
            all_lines = f.read().strip().split("\n")
        for a_line in all_lines:
            orig_token, sim_list = a_line.split(":")
            if orig_token in self.word2idx.keys():
                orig_idx = self.word2idx[orig_token]
                for tok in sim_list:
                    if tok in self.word2idx.keys():
                        self.idx2shapeset[orig_idx].append(self.word2idx[tok])
                        self.idx2shape_freq[orig_idx].append(self.freq_list[self.word2idx[tok]])

    def augment(self, src_tokens: Tensor, sample_dist: Tensor, **unused):
        result = []
        for i, atoken in enumerate(src_tokens.view(-1)):
            tok_idx = int(atoken)
            asub = random.choices(self.idx2shapeset[tok_idx], weights=self.idx2shape_freq[tok_idx], k=1)
            result.append(asub[0])
        final_subs = torch.Tensor(result).to(src_tokens)
        return final_subs * sample_dist.int() + src_tokens * (1 - sample_dist.int())


class PuncAugmentor(BaseAugmentor):
    def __init__(self, text_dict: DICTIONARY_TYPE) -> None:
        super().__init__(text_dict)
        self.punkt_list, self.punkt_freq = [], []
        self.get_punkt_info()

    def get_punkt_info(self):
        punkt_token_list = """，。！？；、（）：“”’‘《》·「」.-,?!~<>()[]【】;:"'—…『』_•〉〈～"""
        for punkt in punkt_token_list:
            p_idx = self.text_dict.index(punkt)
            if p_idx != self.text_dict.unk_index:
                self.punkt_list.append(p_idx)
                self.punkt_freq.append(self.text_dict.count[p_idx])
        return self.punkt_list, self.punkt_freq


class PuncSubAugmentor(PuncAugmentor):
    def augment(self, src_tokens: Tensor, sample_dist: Tensor, **unused):
        result = []
        for i, atoken in enumerate(src_tokens.view(-1)):
            if int(atoken) not in self.punkt_list:
                result.append(atoken)
            else:
                asub = torch.multinomial(torch.Tensor(self.punkt_freq), 1, replacement=True)
                result.append(self.punkt_list[asub])
        sub_tokens = torch.Tensor(result).to(src_tokens)
        return sub_tokens * sample_dist.int() + src_tokens * (1 - sample_dist.int())


class PuncInsAugmentor(PuncAugmentor):
    def __init__(self, text_dict: DICTIONARY_TYPE, ins_delta=0.03) -> None:
        super().__init__(text_dict)
        self.ins_delta = ins_delta

    def augment(self, src_tokens: Tensor, sample_dist: Tensor, **unused):
        padding_mask = src_tokens.ne(self.text_dict.pad())  # mask origanl paddings in a batch
        eos_mask = src_tokens.ne(self.text_dict.eos())  # mask origanl eos in a batch
        total_mask = padding_mask * eos_mask
        result = []
        for i, (atoken, mask) in enumerate(zip(src_tokens.view(-1), total_mask.view(-1))):
            result.append(atoken)
            if int(atoken) not in self.punkt_list:
                ins_or_not = torch.rand((1,))
                if ins_or_not < self.ins_delta and bool(mask):
                    ains = torch.multinomial(torch.Tensor(self.punkt_freq), 1, replacement=True)
                    result.append(self.punkt_list[ains])
        ins_tokens = torch.Tensor(result).to(src_tokens)
        return ins_tokens


class PuncDelAugmentor(PuncAugmentor):
    def __init__(self, text_dict: DICTIONARY_TYPE, del_delta=0.5) -> None:
        super().__init__(text_dict)
        self.del_delta = del_delta

    def augment(self, src_tokens: Tensor, sample_dist: Tensor, **unused):
        padding_mask = src_tokens.eq(self.text_dict.pad())  # mask origanl paddings in a batch
        bos_mask = src_tokens.eq(self.text_dict.bos())  # mask origanl bos in a batch
        eos_mask = src_tokens.eq(self.text_dict.eos())  # mask origanl eos in a batch
        punkt_mask = []
        for i, atoken in enumerate(src_tokens.view(-1)):
            if int(atoken) not in self.punkt_list:
                punkt_mask.append(True)
            else:
                punkt_mask.append(False)
        punkt_mask = torch.BoolTensor(punkt_mask)
        sample_dist_for_keep = (1 - torch.bernoulli(self.del_delta + torch.zeros_like(src_tokens))).bool()
        sample_dist_for_keep = sample_dist_for_keep | padding_mask | bos_mask | eos_mask | punkt_mask
        return src_tokens[sample_dist_for_keep.nonzero()].view(-1)


class MixAugmentor(BaseAugmentor):
    def __init__(
            self, text_dict: DICTIONARY_TYPE,
            augment_setting: Dict[str, Any],
    ) -> None:
        super().__init__(text_dict)
        self.augment_mode = augment_setting["augment_mode"]
        self.num_modes = len(self.augment_mode)
        self.augment_setting = augment_setting
        self.shape_file = augment_setting["shape_file"] if "shape_file" in augment_setting.keys() else None
        self.fuzzy_thre = augment_setting["fuzzy_threshold"] if "fuzzy_threshold" in augment_setting.keys() else 1
        self.augmentors = []
        self.fuzzy = ('fuzzy_pinyin' in self.augment_mode)
        # ipdb.set_trace()
        self._build_augmentors()

    def _build_augmentors(self):
        modes_options = ['mask', 'random', 'freq', 'pinyin', 'fuzzy_pinyin' 'shape', 'delete', 'insert']
        for mode in self.augment_mode:
            if mode == 'mask':
                self.augmentors.append(MaskAugmentor(self.text_dict))
            elif mode == 'random':
                # TODO: we should have a better method to find out the num of special tokens
                self.augmentors.append(RandomAugmentor(self.text_dict, special_token_num=5))
            elif mode == 'freq':
                self.augmentors.append(WordFreqAugmentor(self.text_dict))
            elif mode == 'pinyin':
                self.augmentors.append(PinyinAugmentor(self.text_dict))
            elif mode == 'fuzzy_pinyin':
                self.augmentors.append(PinyinAugmentor(self.text_dict, fuzzy=self.fuzzy, fuzzy_thre=self.fuzzy_thre))
            elif mode == 'shape':
                self.augmentors.append(ShapeAugmentor(self.text_dict, self.shape_file))
            elif mode == 'delete':
                self.augmentors.append(RandDelAugmentor(self.text_dict, delta=self.augment_setting['probs'][
                    'delete'] if 'delete' in self.augment_setting.keys() else 0))
            elif mode == 'insert':
                self.augmentors.append(RandInsAugmentor(self.text_dict, delta=self.augment_setting['probs'][
                    'insert'] if 'insert' in self.augment_setting.keys() else 0))
            else:
                raise NotImplementedError

    def augment(self, src_tokens: Tensor, sample_dist: Tensor, **unused):
        rand_mode = random.choice(range(self.num_modes))
        return self.augmentors[rand_mode].augment(src_tokens, sample_dist, **unused)


class PuncMixAugmentor(BaseAugmentor):
    def __init__(
            self, text_dict: DICTIONARY_TYPE,
            augment_setting: Dict[str, Any],
            masker: MaskScheme
    ) -> None:
        super().__init__(text_dict)
        self.augment_mode = augment_setting["punc_augment"]['mode']
        self.num_modes = len(self.augment_mode)
        self.punc_augment_choice = augment_setting["punc_augment"]['choice']
        self.augment_setting = augment_setting
        self.augmentors = {}
        self._build_augmentors()
        self.mode_options = ['punc_del', 'punc_sub', 'punc_ins']
        self.masker = masker

    def _build_augmentors(self):
        for mode in self.augment_mode:
            if mode == 'punc_del':
                self.augmentors['punc_del'] = PuncDelAugmentor(self.text_dict,
                                                               del_delta=self.augment_setting['punc_augment'][
                                                                   'del_delta'])
            elif mode == 'punc_sub':
                self.augmentors['punc_sub'] = PuncSubAugmentor(self.text_dict)
            elif mode == 'punc_ins':
                self.augmentors['punc_ins'] = PuncInsAugmentor(self.text_dict,
                                                               ins_delta=self.augment_setting['punc_augment'][
                                                                   'ins_delta'])
            else:
                raise NotImplementedError

    def augment(self, src_tokens: Tensor, sample_dist: Tensor, **unused):
        if self.punc_augment_choice == "random":
            rand_mode = random.choice(self.augment_mode)
            return self.augmentors[rand_mode].augment(src_tokens, sample_dist, **unused)
        elif self.punc_augment_choice == "mix":
            for mode in self.mode_options:
                if mode in self.augmentors.keys():
                    sample_dist = self.masker.mask(src_tokens, self.text_dict)
                    src_tokens = self.augmentors[mode].augment(src_tokens, sample_dist, **unused)
            return src_tokens


class AugmentorWrapper(object):
    def __init__(
            self, text_dict: DICTIONARY_TYPE,
            augment_setting: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.augment_setting = augment_setting
        self.augmentor = MixAugmentor(text_dict, augment_setting)
        self.with_punc = False
        self.masker = MaskScheme(
            delta=augment_setting['probs']['substitute'],
            p=augment_setting['span_mask_p'] if 'span_mask_p' in self.augment_setting.keys() else 0.2,
            span_max_len=augment_setting['span_max_len'] if 'span_max_len' in self.augment_setting.keys() else 5,
            span_min_len=augment_setting['span_min_len'] if 'span_min_len' in self.augment_setting.keys() else 1,
        )
        if "punc_augment" in self.augment_setting.keys():
            self.with_punc = True
            self.punc_masker = MaskScheme(
                delta=augment_setting['punc_augment']['sub_delta'],
            )
            self.punc_augmentor = PuncMixAugmentor(text_dict, augment_setting, self.punc_masker)

    def sample_dist(self, src_tokens: Tensor):
        if self.augment_setting['mask_level'] == 'char':
            mask_sample = self.masker.mask(src_tokens, self.augmentor.text_dict)
        elif self.augment_setting['mask_level'] == 'span':
            mask_sample = self.masker.span_mask(src_tokens, self.augmentor.text_dict)
        else:
            raise NotImplementedError
        return mask_sample

    def augment(self, src_tokens: Tensor):
        mask_sample = self.sample_dist(src_tokens)
        if self.with_punc:
            src_tokens = self.punc_augmentor.augment(src_tokens, mask_sample)
            mask_sample = self.sample_dist(src_tokens)
        return self.augmentor.augment(src_tokens, mask_sample)
