import os
from shutil import copyfile

import sentencepiece as spm

from transformers.tokenization_utils import PreTrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.unigram.model"}

class MFBERTTokenizer(PreTrainedTokenizer):
    """
        Adapted from RobertaTokenizer and XLNetTokenizer
        SentencePiece based tokenizer. Peculiarities:

            - requires `SentencePiece <https://github.com/google/sentencepiece>`_
    """

    vocab_files_names = VOCAB_FILES_NAMES
    


    def __init__(
        self,
        vocab_file,
        dict_file,
        bos_token="[CLS]",
        eos_token="[SEP]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=["[CLS]", "[SEP]"],
        **kwargs
    ):
        with open(dict_file, 'r') as f:
            self.dict = []
            for line in f:
                self.dict.append(line.split()[0])
        self.fairseq_tokens_to_ids = {"[CLS]": 0, "<pad>": 1, "[SEP]": 2, "<unk>": 3}
        self.fairseq_offset = len(self.fairseq_tokens_to_ids)
        self.fairseq_tokens_to_ids["<mask>"] = len(self.dict) + len(self.fairseq_tokens_to_ids)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}


        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        
# #         self.model_max_length = int(1e30)
#         self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
#         self.max_len_sentences_pair = self.max_len - 4  # take into account special tokens
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file
        # HACK: These tokens were added by fairseq but don't seem to be actually used when duplicated in the actual
        # sentencepiece vocabulary (this is the case for [CLS] and [SEP]
        self.fairseq_tokens_to_ids = {"[CLS]": 0, "<pad>": 1, "[SEP]": 2, "<unk>": 3}
        self.fairseq_offset = len(self.fairseq_tokens_to_ids)
        self.fairseq_tokens_to_ids["<mask>"] = len(self.dict) + len(self.fairseq_tokens_to_ids)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}


    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:
            single sequence: [CLS] X [SEP]
            pair of sequences: [CLS] A [SEP][SEP] B [SEP]
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep


    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]


    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A RoBERTa sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep) * [0] + len(token_ids_1 + sep) * [1]


    @property
    def vocab_size(self):
        return len(self.fairseq_tokens_to_ids) + len(self.dict)

    def _tokenize(self, text):
        return self.sp_model.EncodeAsPieces(text)
    
    def PieceToId(self, piece):
        if piece in self.dict:
            return self.dict.index(piece)
        return -1
    
    def IdToPiece(self, index):
        return self.dict[index]

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        elif self.PieceToId(token) == -1:
            # Convert sentence piece unk token to fairseq unk token index
            return self.unk_token_id
        return self.fairseq_offset + self.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.IdToPiece(index - self.fairseq_offset)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use AlbertTokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece"
            )
            raise
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        for token in tokens:
            if token in self.fairseq_tokens_to_ids:
                tokens.remove(token)
            
        out_string = "".join(tokens).replace('▁', " ").strip()
        return out_string


    def save_vocabulary(self, save_directory, filename_prefix):
        """ Save the sentencepiece vocabulary (copy original file) and special tokens file
            to a directory.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)