import json
import re

class Tokenizer:
    def __init__(self):
        self.special_tokens = ["<PAD>","<UNK>","<BOS>","<EOS>",  ]
        self.word2id = {
            word: idx
            for idx , word in enumerate(self.special_tokens)
        }
        self.id2word = {
            idx: word
            for word, idx in self.word2id.items()
        }

    def fit(self, text):
        words = re.findall(r"\w+|[^\w\s]", text)

        vocab = sorted(set(words))

        # self.word2id = {
        #     word: idx
        #     for idx , word in enumerate(vocab)
        # }
        next_id = len(self.special_tokens)
        
        for word in vocab:
            if word not in self.word2id:
                self.word2id[word] = next_id
                next_id += 1

        self.id2word = {
            idx: word
            for word, idx in self.word2id.items()
        }

    def encode(self, text):
        words = re.findall(r"\w+|[^\w\s]", text)

        ids = [self.word2id["<BOS>"]]
        print(text)
        for word in words:
            if word not in self.word2id:
                new_id = len(self.word2id)
                self.word2id[word] = new_id
                self.id2word[new_id] = word
            
            ids.append(self.word2id[word])
        
        ids.append(self.word2id["<EOS>"])
        print(ids)
        return ids
    
    def decode(self, ids):
        words = []
        for i in ids:
            token = self.id2word[i]
            print(token)
            if token in ["<PAD>", "<BOS>", "<EOS>"]:
                continue

            words.append(token)
        print(words)
        return " ".join(words)
    
    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.word2id, f, indent=4)
    
    def load(self, path):
        with open(path) as f:
            self.word2id = json.load(f)

        self.id2word = {
            int(v): k
            for k, v in self.word2id.items()
        }
