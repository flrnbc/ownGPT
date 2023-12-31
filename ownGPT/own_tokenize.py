from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
#from config import Config
from typing import List


def postprocess_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    # postprocessing a la BERT and transformer?!?
    # NOTE: the ordering is important
    # TODO: is this really correct? still quite unclear...
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    return tokenizer


def train_BPE_tokenizer(
    file_paths: List[Path], tokens_save_path: Path = None
) -> Tokenizer:
    # TODO: fix file_paths -> either List[str] or cast here
    # TODO: need unknown token?!
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[BOS]", "[EOS]", "[MSK]"],
        #vocab_size=Config.vocab_size,
    )
    tokenizer.pre_tokenizer = Whitespace()
    # TODO: already add convert to string here?
    tokenizer.train(file_paths, trainer)
    if tokens_save_path:
        tokenizer.save(tokens_save_path)
    # post processing
    # NOTE: the ordering is important
    # TODO: is this correct? still quite unclear...
    tokenizer = postprocess_tokenizer(tokenizer)
    return tokenizer


def load_BPE_tokenizer(tokens_file: Path, postprocess: bool = True) -> Tokenizer:
    tokenizer = Tokenizer.from_file(tokens_file)
    if postprocess:
        tokenizer = postprocess_tokenizer(tokenizer)
    return tokenizer


def encode_file(tokenizer: Tokenizer, file: Path , limit: int=None) -> str:
    with open(file, 'r') as f: # TODO: 'rb'?
        text = f.read()
    if limit:
        if limit > len(text) - 1:
            limit = len(text) - 1
        text = text[:limit]
    return tokenizer.encode(text)





# NOTE: need conversion to string because of PyString (used in C++ extensions?!?) maybe change above
#file_path = [str(file) for file in config.tokenize_path.glob("*.txt")]
#tokenizer = train_BPE_tokenizer(file_path, tokens_save_path=str(config.tokens_path / Path("test_tokens.json")))
#print(tokenizer.encode("Hello, my lord!", "How is it going?").ids)
