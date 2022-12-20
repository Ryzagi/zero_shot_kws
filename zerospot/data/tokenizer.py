from pathlib import Path
import json


class Tokenizer:
    def __init__(self):
        self.tokens = json.loads((Path(__file__).parent / 'tokens.json').read_text())
        self.pad_token_id = 0

    def __call__(self, word_list):
        tokens_list_ids = []
        tokens_list_lentghts = []
        max_length = max([len(str(word)) for word in word_list])
        for word in word_list:
            tokens_ids = []
            tokens_lentghts = []
            for char in word:
                token_id = self.tokens[char]
                tokens_ids.append(token_id)
            token_length = len(tokens_ids)
            padding = [self.pad_token_id] * (max_length - token_length)
            tokens_ids.extend(padding)
            tokens_lentghts.append(token_length)
            tokens_list_ids.append(tokens_ids)
            tokens_list_lentghts.extend(tokens_lentghts)
        return tokens_list_ids, tokens_list_lentghts
