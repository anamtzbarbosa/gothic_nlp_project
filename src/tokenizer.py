"""
Byte Pair Encoding (BPE) inspired by 'minbpe'.
This script trains a tokenizer on the Gothic Literature dataset, 
encodes the corpus into integers, and the saves 
both the model rules (JSON) and the pre tokenized data (Pickle).
"""
import json
from tqdm import tqdm
import pickle

class GothicBPE:
    def __init__(self):
        self.merges = {} # dicc -> which 2 characters (idx) merged into another token
                        # {(104, 101): 256}
        self.vocab = {}  # ID -> Token

    def counts(self, ids):
        """Counts how many times a pair of tokens appear together """
        counts = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1]) 
            # If pair already in dicc + 1
            if pair in counts:
                counts[pair] = counts[pair] + 1
            # Otherwise create new element
            else:
                counts[pair] = 1

        return counts

    def merge(self, ids, pair, idx):
        """ Merge 2 dif tokens in text into 1 new (new ID as well)
            Ex: ids = [1, 2, 3, 1, 2] pair = (1,2) idx = 99 
            new_ids = [99,3,99]
        """
        new_ids = []
        i = 0
        while i < len(ids):
            # If not last element 
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2 # Since we merged 2 tokes, we skip 2
            else: #if not number we looking for, remain the same id
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def build_vocab(self):
        vocab =  {}
        for i in range(256): #individual bytes
            vocab[i] = bytes([i]) # {65: b'A', 66: b'B'}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab
    
    def train(self, text, vocab_size):
        tokens = list(text.encode("utf-8")) # text into bytes (0-255) -> important since we have characters from other languages - if vocabulary unicode -> infinite
        ids = list(tokens)
        num_merges = vocab_size - 256 # we already hace 256 dif tokens (0-255)   
        
        print(f"Training BPE - vocabulariy size: {vocab_size}...")
        for i in tqdm(range(num_merges), desc="Merging"):
            # count frequent pair tokens
            stats = self.counts(ids)
            # most frequent pair = priority to merge
            pair = max(stats, key=stats.get)
            # New id for new pair
            idx = 256 + i
            # Merge and replace new token in text 
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx #save
        
        self.vocab = self.build_vocab()
    

    def encode(self, text):
        """ Encode text into list num with our trained tokenizer"""
        tokens = list(text.encode("utf-8"))

        # We merge in order, first pair found is merged first (merfe is in order)
        for pair, new_id in self.merges.items():
            # Each time you see the rule replace it -> EX:  (104, 111), change it for 256
            tokens = self.merge(tokens, pair, new_id)
            
            # Early exit: If after rule only 1 token left, finish
            if len(tokens) < 2:
                break
            
        return tokens
    
    def decode(self, ids):
        """ From numers to text"""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8") 

    def save_model(self, path):
        """Saves the merge rules """
        # Convert tuple keys to strings
        s_merges = {f"{p[0]},{p[1]}": idx for p, idx in self.merges.items()}
        data = {"merges": s_merges, "vocab_size": len(self.vocab) if self.vocab else (len(self.merges) + 256)}
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Tokenizer model saved to {path}")
    
if __name__ == "__main__":  
    tokenizer = GothicBPE()
    with open('data/corpus_clean.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer.train(text, vocab_size=5000)

    # Save tokenizer and encoded text
    tokenizer.save_model('data/gothic_tokenizer.json') # {"merges": {...: ..., ...:..., } {"vocab_size" : 1000}}

    print("Tokenizing corpus...")
    tokens_corpus = tokenizer.encode(text) 
    with open('data/corpus_tokenized.pkl', 'wb') as f:
        pickle.dump(tokens_corpus, f) #binary

    print(f"Tokenized corpus saved ({len(tokens_corpus)} tokens)")