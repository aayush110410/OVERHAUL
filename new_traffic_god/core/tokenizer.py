"""
Traffic-Specific Tokenizer
==========================

Custom tokenizer that understands:
- Standard natural language
- Traffic terminology (congestion, gridlock, etc.)
- Location names (Noida sectors, Indirapuram, NCR)
- Numeric traffic data
- Temporal expressions
- Road names and landmarks
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
import math


class TrafficTokenizer:
    """
    Tokenizer with traffic-domain awareness
    
    Features:
    - BPE subword tokenization
    - Special tokens for traffic concepts
    - Number tokenization (for traffic counts, speeds, etc.)
    - Location-aware tokenization
    - Temporal expression handling
    """
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    SEP_TOKEN = "<SEP>"
    
    # Traffic-specific special tokens
    TIME_TOKEN = "<TIME>"
    LOCATION_TOKEN = "<LOC>"
    SPEED_TOKEN = "<SPEED>"
    CONGESTION_TOKEN = "<CONG>"
    ROUTE_TOKEN = "<ROUTE>"
    PREDICTION_TOKEN = "<PRED>"
    QUERY_TOKEN = "<QUERY>"
    RESPONSE_TOKEN = "<RESP>"
    
    def __init__(
        self,
        vocab_size: int = 50257,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Core vocabulary
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # BPE merges
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        
        # Special tokens
        self.special_tokens = [
            self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN,
            self.SEP_TOKEN, self.TIME_TOKEN, self.LOCATION_TOKEN, self.SPEED_TOKEN,
            self.CONGESTION_TOKEN, self.ROUTE_TOKEN, self.PREDICTION_TOKEN,
            self.QUERY_TOKEN, self.RESPONSE_TOKEN
        ]
        
        if special_tokens:
            self.special_tokens.extend(special_tokens)
        
        # Traffic domain vocabulary (Noida/NCR specific)
        self.traffic_vocab = self._build_traffic_vocabulary()
        
        # Location patterns
        self.location_patterns = self._build_location_patterns()
        
        # Initialize base vocabulary
        self._initialize_vocabulary()
        
        # Regex patterns
        self.number_pattern = re.compile(r'\d+\.?\d*')
        self.time_pattern = re.compile(r'\d{1,2}:\d{2}(?:\s*(?:AM|PM|am|pm))?')
        self.speed_pattern = re.compile(r'\d+\.?\d*\s*(?:km/h|kmph|mph|kph)', re.IGNORECASE)
    
    def _build_traffic_vocabulary(self) -> Dict[str, str]:
        """Build traffic-specific vocabulary"""
        return {
            # Traffic states
            "congestion": "traffic_state",
            "gridlock": "traffic_state",
            "jam": "traffic_state",
            "bottleneck": "traffic_state",
            "freeflow": "traffic_state",
            "heavy": "traffic_intensity",
            "moderate": "traffic_intensity",
            "light": "traffic_intensity",
            
            # Road types
            "highway": "road_type",
            "expressway": "road_type",
            "arterial": "road_type",
            "flyover": "road_type",
            "underpass": "road_type",
            "roundabout": "road_type",
            "intersection": "road_type",
            "signal": "road_type",
            
            # Time expressions
            "rush hour": "time_period",
            "peak hour": "time_period",
            "morning rush": "time_period",
            "evening rush": "time_period",
            "off-peak": "time_period",
            
            # Actions
            "commute": "action",
            "travel": "action",
            "route": "action",
            "navigate": "action",
            "avoid": "action",
            
            # Metrics
            "ETA": "metric",
            "travel time": "metric",
            "delay": "metric",
            "speed": "metric",
            "flow": "metric",
            "density": "metric",
            "AQI": "metric",
            
            # Infrastructure
            "metro": "infrastructure",
            "bus": "infrastructure",
            "parking": "infrastructure",
            "toll": "infrastructure",
            "booth": "infrastructure"
        }
    
    def _build_location_patterns(self) -> Dict[str, List[str]]:
        """Build Noida/NCR location vocabulary"""
        return {
            "noida_sectors": [f"Sector {i}" for i in range(1, 169)] + 
                           [f"Sector-{i}" for i in range(1, 169)] +
                           [f"Sec {i}" for i in range(1, 169)] +
                           [f"Sec-{i}" for i in range(1, 169)],
            
            "noida_landmarks": [
                "Noida City Centre", "Botanical Garden", "Golf Course",
                "Film City", "Sector 18", "DND Flyway", "Noida Expressway",
                "Greater Noida Expressway", "Electronic City", "Noida Stadium",
                "Amity University", "Jaypee Hospital", "Fortis Hospital",
                "DLF Mall", "GIP Mall", "Logix Mall", "Centre Stage Mall",
                "Wave City", "Pari Chowk", "Alpha 1", "Alpha 2", "Beta 1", "Beta 2",
                "Gamma 1", "Gamma 2", "Delta", "Sigma", "Zeta", "Eta", "Phi",
                "Chi", "Omega", "Knowledge Park", "Techzone"
            ],
            
            "indirapuram": [
                "Indirapuram", "Vaishali", "Kaushambi", "Vasundhara",
                "Shakti Khand", "Niti Khand", "Gyan Khand", "Ahinsa Khand",
                "Nyay Khand", "Abhay Khand", "Shipra Mall", "Mahagun Mall",
                "Shipra Suncity", "Jaipuria Mall"
            ],
            
            "ghaziabad": [
                "Ghaziabad", "Raj Nagar Extension", "Crossing Republik",
                "Mohan Nagar", "Kavi Nagar", "Nehru Nagar", "Sahibabad",
                "Lal Kuan", "Ghaziabad Junction", "NH24", "NH9"
            ],
            
            "delhi_ncr": [
                "Delhi", "New Delhi", "Connaught Place", "India Gate",
                "Akshardham", "Mayur Vihar", "Laxmi Nagar", "Preet Vihar",
                "Anand Vihar", "ISBT", "Kashmere Gate", "ITO",
                "Yamuna Bank", "Nizamuddin", "Sarai Kale Khan",
                "Dwarka", "Gurugram", "Gurgaon", "Faridabad"
            ],
            
            "major_roads": [
                "NH24", "NH9", "NH19", "NH44", "DND Flyway", "Noida Expressway",
                "Greater Noida Expressway", "Yamuna Expressway", "Eastern Peripheral",
                "Western Peripheral", "Dadri Road", "Ghaziabad Link Road"
            ]
        }
    
    def _initialize_vocabulary(self):
        """Initialize the base vocabulary"""
        idx = 0
        
        # Add special tokens first
        for token in self.special_tokens:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        # Add traffic vocabulary
        for term in self.traffic_vocab.keys():
            if term not in self.token_to_id:
                self.token_to_id[term] = idx
                self.id_to_token[idx] = term
                idx += 1
        
        # Add location vocabulary
        for category, locations in self.location_patterns.items():
            for loc in locations:
                if loc.lower() not in self.token_to_id:
                    self.token_to_id[loc.lower()] = idx
                    self.id_to_token[idx] = loc.lower()
                    idx += 1
        
        # Add basic ASCII characters
        for i in range(256):
            char = chr(i)
            if char not in self.token_to_id:
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
                idx += 1
        
        # Add number tokens
        for i in range(1000):
            num_token = f"<NUM_{i}>"
            self.token_to_id[num_token] = idx
            self.id_to_token[idx] = num_token
            idx += 1
        
        self._current_vocab_size = idx
    
    def _get_pairs(self, word: List[str]) -> set:
        """Get all adjacent pairs in a word"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _bpe(self, token: str) -> List[str]:
        """Apply BPE to a single token"""
        word = list(token)
        
        if len(word) == 1:
            return word
        
        while True:
            pairs = self._get_pairs(word)
            if not pairs:
                break
            
            # Find the pair with lowest rank (highest priority)
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
            
            if bigram not in self.bpe_ranks:
                break
            
            # Merge the pair
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
        
        return word
    
    def train(self, texts: List[str], num_merges: int = 10000):
        """
        Train the BPE tokenizer on a corpus
        
        Args:
            texts: List of training texts
            num_merges: Number of BPE merges to learn
        """
        # Count word frequencies
        word_freqs = defaultdict(int)
        
        for text in texts:
            # Pre-tokenize
            words = self._pre_tokenize(text)
            for word in words:
                word_freqs[tuple(word)] += 1
        
        # Learn BPE merges
        for i in range(num_merges):
            # Count pairs
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = list(word)
                for j in range(len(symbols) - 1):
                    pair = (symbols[j], symbols[j + 1])
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                break
            
            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # Add to vocabulary
            self.bpe_ranks[best_pair] = i
            new_token = best_pair[0] + best_pair[1]
            
            if new_token not in self.token_to_id:
                idx = self._current_vocab_size
                self.token_to_id[new_token] = idx
                self.id_to_token[idx] = new_token
                self._current_vocab_size += 1
            
            # Merge in vocabulary
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq
            
            word_freqs = new_word_freqs
            
            if (i + 1) % 1000 == 0:
                print(f"BPE merge {i + 1}/{num_merges}: {best_pair} -> {new_token}")
    
    def _pre_tokenize(self, text: str) -> List[List[str]]:
        """Pre-tokenize text into words"""
        # Handle special patterns first
        text = self._handle_special_patterns(text)
        
        # Split on whitespace and punctuation
        pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        try:
            import regex
            tokens = regex.findall(pattern, text)
        except ImportError:
            # Fallback to simple splitting
            tokens = re.findall(r'\w+|[^\w\s]', text)
        
        return [list(token) for token in tokens if token.strip()]
    
    def _handle_special_patterns(self, text: str) -> str:
        """Handle traffic-specific patterns"""
        # Replace times with special token
        text = self.time_pattern.sub(lambda m: f" {self.TIME_TOKEN}{m.group()}{self.TIME_TOKEN} ", text)
        
        # Replace speeds with special token
        text = self.speed_pattern.sub(lambda m: f" {self.SPEED_TOKEN}{m.group()}{self.SPEED_TOKEN} ", text)
        
        return text
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False
    ) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
        
        Returns:
            List of token IDs
        """
        # Pre-tokenize
        words = self._pre_tokenize(text.lower())
        
        # Apply BPE and convert to IDs
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.token_to_id[self.BOS_TOKEN])
        
        for word in words:
            # Check if it's a known traffic term
            word_str = ''.join(word)
            if word_str in self.token_to_id:
                token_ids.append(self.token_to_id[word_str])
            else:
                # Apply BPE
                subwords = self._bpe(''.join(word))
                for subword in subwords:
                    if subword in self.token_to_id:
                        token_ids.append(self.token_to_id[subword])
                    else:
                        # Character-level fallback
                        for char in subword:
                            token_ids.append(self.token_to_id.get(char, self.token_to_id[self.UNK_TOKEN]))
        
        if add_special_tokens:
            token_ids.append(self.token_to_id[self.EOS_TOKEN])
        
        # Handle length constraints
        if max_length is not None:
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                if add_special_tokens:
                    token_ids[-1] = self.token_to_id[self.EOS_TOKEN]
            
            if padding and len(token_ids) < max_length:
                pad_length = max_length - len(token_ids)
                token_ids.extend([self.token_to_id[self.PAD_TOKEN]] * pad_length)
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.UNK_TOKEN)
            
            if skip_special_tokens and token in self.special_tokens:
                continue
            
            tokens.append(token)
        
        # Join and clean up
        text = ''.join(tokens)
        text = text.replace('Ġ', ' ')  # Handle GPT-2 style space encoding
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, List[List[int]]]:
        """
        Encode a batch of texts
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        encoded = [self.encode(text, max_length=max_length, truncation=truncation) for text in texts]
        
        # Find max length for padding
        if padding:
            if max_length is None:
                max_length = max(len(ids) for ids in encoded)
            
            attention_masks = []
            for i, ids in enumerate(encoded):
                mask = [1] * len(ids)
                if len(ids) < max_length:
                    pad_length = max_length - len(ids)
                    ids.extend([self.token_to_id[self.PAD_TOKEN]] * pad_length)
                    mask.extend([0] * pad_length)
                encoded[i] = ids
                attention_masks.append(mask)
        else:
            attention_masks = [[1] * len(ids) for ids in encoded]
        
        return {
            "input_ids": encoded,
            "attention_mask": attention_masks
        }
    
    def save(self, path: str):
        """Save tokenizer to file"""
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "bpe_ranks": {f"{k[0]}|||{k[1]}": v for k, v in self.bpe_ranks.items()},
            "special_tokens": self.special_tokens,
            "traffic_vocab": self.traffic_vocab
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> "TrafficTokenizer":
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in enumerate(data["token_to_id"].keys())}
        tokenizer.bpe_ranks = {tuple(k.split("|||")): v for k, v in data["bpe_ranks"].items()}
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.traffic_vocab = data["traffic_vocab"]
        
        return tokenizer
    
    @property
    def vocab_size_actual(self) -> int:
        return len(self.token_to_id)
    
    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]
    
    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]
    
    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]


# Quick test
if __name__ == "__main__":
    tokenizer = TrafficTokenizer()
    
    test_texts = [
        "What is the traffic like on Noida Expressway at 9:00 AM?",
        "Best route from Sector 62 to Indirapuram avoiding congestion",
        "Current AQI in Noida and how it affects traffic flow",
        "Predict traffic on NH24 for tomorrow morning rush hour"
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Original: {text}")
        print(f"Tokens: {len(encoded)}")
        print(f"Decoded: {decoded}")
        print()
