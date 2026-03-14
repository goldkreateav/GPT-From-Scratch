import re

class SimpleTokenizerV2:
    def __init__(self, vocab = {}):
        self.str_to_int = vocab
        self.int_to_str = {integer: string for string, integer in vocab}

    def encode(self, text):
        tokens = re.split(r"""([.,!?:'"-()\\_;]|\s|[--]|\n)""", text)
        tokens = [r for r in tokens if r != " " and r != ""]
        return [self.str_to_int[t] if t in self.str_to_int else self.str_to_int["<|unk|>"] for t in tokens]

    def decode(self, encoded):
        return [self.int_to_str[n] for n in encoded]

    def get_vocab_from_text(self, text):
        tokens = re.split(r"""([.,!?:'"-()\\_;]|\s|[--]|\n)""", text)
        tokens = [r for r in tokens if r != " " and r != ""]
        tokens.extend(["<|endoftext|>", "<|unk|>"])
        dictionary = {}
        for token in tokens:
            if token in dictionary:
                dictionary[token] += 1
            else:
                dictionary[token] = 1
        tokens_sorted = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True)).keys()
        vocab = {token:integer for integer, token in enumerate(tokens_sorted)}
        self.str_to_int = vocab
        self.int_to_str = {integer:string for string, integer in vocab.items()}