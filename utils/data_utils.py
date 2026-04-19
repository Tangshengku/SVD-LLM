import os
import random
import torch
import sys
from datasets import load_dataset
from torch.utils.data.dataset import Dataset

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


EVOL_CODEALPACA_DATASET = "theblackcat102/evol-codealpaca-v1"
TULU_MATH_DATASET = "allenai/tulu-3-sft-personas-math"


def _cache_safe_name(name):
    return name.replace("/", "_")


def _is_evol_codealpaca(name):
    normalized = name.lower()
    return normalized in {
        "evol-codealpaca",
        "evol-codealpaca-v1",
        EVOL_CODEALPACA_DATASET.lower(),
    }


def _is_tulu_math(name):
    normalized = name.lower()
    return normalized in {
        "tulu-math",
        "personas-math",
        "tulu-3-sft-personas-math",
        TULU_MATH_DATASET.lower(),
    }


def _is_mixture_dataset(name):
    return name.startswith("mix:")


def _split_mixture_dataset(name):
    if not _is_mixture_dataset(name):
        return [name]
    parts = [part.strip() for part in name[4:].split(",") if part.strip()]
    if not parts:
        raise ValueError("Mixture dataset spec must include at least one dataset, e.g. mix:wikitext2,evol-codealpaca,tulu-math")
    return parts


def _allocate_mixture_counts(total, n_sources):
    if n_sources <= 0:
        raise ValueError("n_sources must be positive")
    base = total // n_sources
    remainder = total % n_sources
    return [base + (1 if idx < remainder else 0) for idx in range(n_sources)]


def _format_instruction_sample(sample):
    parts = []
    instruction = sample.get("instruction")
    input_text = sample.get("input")
    output = sample.get("output")

    if instruction:
        parts.append(f"Instruction:\n{instruction.strip()}")
    if input_text:
        parts.append(f"Input:\n{input_text.strip()}")
    if output:
        parts.append(f"Response:\n{output.strip()}")
    if not parts:
        raise ValueError("Unsupported sample format for instruction dataset.")
    return "\n\n".join(parts)


def _format_chat_messages(messages):
    formatted = []
    for message in messages:
        role = message.get("role", "unknown").strip().title()
        content = message.get("content", "").strip()
        if content:
            formatted.append(f"{role}:\n{content}")
    return "\n\n".join(formatted)


def _format_math_sample(sample):
    parts = []
    prompt = sample.get("prompt")
    if prompt:
        parts.append(f"Prompt:\n{prompt.strip()}")
    messages = sample.get("messages")
    if messages:
        formatted_messages = _format_chat_messages(messages)
        if formatted_messages:
            parts.append(formatted_messages)
    if not parts:
        raise ValueError("Unsupported sample format for math dataset.")
    return "\n\n".join(parts)


def _load_training_texts(name, dataset_cache_dir=None):
    if _is_mixture_dataset(name):
        texts = []
        for dataset_name in _split_mixture_dataset(name):
            texts.extend(_load_training_texts(dataset_name, dataset_cache_dir))
        return texts
    if name == "c4":
        traindata = load_dataset("json", data_files="utils/c4-train.json")["train"]
        return list(traindata["text"])
    if name == "ptb":
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train", cache_dir=dataset_cache_dir)
        return list(traindata["sentence"])
    if name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=dataset_cache_dir)
        return list(traindata["text"])
    if _is_evol_codealpaca(name):
        traindata = load_dataset(EVOL_CODEALPACA_DATASET, split="train", cache_dir=dataset_cache_dir)
        return [_format_instruction_sample(sample) for sample in traindata]
    if _is_tulu_math(name):
        traindata = load_dataset(TULU_MATH_DATASET, split="train", cache_dir=dataset_cache_dir)
        return [_format_math_sample(sample) for sample in traindata]
    raise NotImplementedError(f"Unsupported dataset: {name}")


def _build_calibration_chunks_from_texts(texts, tokenizer, nsamples, seqlen, seed, batch_size):
    random.seed(seed)
    tot_text = "\n\n".join(texts)
    traindataset = []
    total_steps = nsamples + 1
    pending_batch = None
    pending_count = 0

    for _ in range(total_steps):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        current = trainenc.input_ids[:, :seqlen]
        if pending_batch is None:
            pending_batch = current
            pending_count = 1
        else:
            pending_batch = torch.cat((pending_batch, current), dim=0)
            pending_count += 1
        if pending_count == batch_size:
            attention_mask = torch.ones_like(pending_batch)
            traindataset.append({"input_ids": pending_batch, "attention_mask": attention_mask})
            pending_batch = None
            pending_count = 0

    if pending_batch is not None:
        attention_mask = torch.ones_like(pending_batch)
        traindataset.append({"input_ids": pending_batch, "attention_mask": attention_mask})
    return traindataset


def _sample_loader_from_texts(texts, nsamples, seed, seqlen, tokenizer):
    trainenc = tokenizer("\n\n".join(texts), return_tensors="pt")
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def _build_mixture_calibration_data(name, tokenizer, nsamples, seqlen, seed, batch_size, dataset_cache_dir=None):
    dataset_names = _split_mixture_dataset(name)
    counts = _allocate_mixture_counts(nsamples, len(dataset_names))
    mixed = []
    for idx, (dataset_name, count) in enumerate(zip(dataset_names, counts)):
        if count <= 0:
            continue
        texts = _load_training_texts(dataset_name, dataset_cache_dir)
        mixed.extend(
            _build_calibration_chunks_from_texts(
                texts=texts,
                tokenizer=tokenizer,
                nsamples=count,
                seqlen=seqlen,
                seed=seed + idx,
                batch_size=batch_size,
            )
        )
    return mixed


def _build_mixture_loader(name, nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    dataset_names = _split_mixture_dataset(name)
    counts = _allocate_mixture_counts(nsamples, len(dataset_names))
    trainloader = []
    for idx, (dataset_name, count) in enumerate(zip(dataset_names, counts)):
        if count <= 0:
            continue
        texts = _load_training_texts(dataset_name, dataset_cache_dir)
        trainloader.extend(
            _sample_loader_from_texts(
                texts=texts,
                nsamples=count,
                seed=seed + idx,
                seqlen=seqlen,
                tokenizer=tokenizer,
            )
        )
    return trainloader

def get_calib_train_data(name, tokenizer, nsamples, seqlen=2048, seed=3, batch_size=1, dataset_cache_dir=None):
    import random
    random.seed(seed)
    cache_file = (
        f"cache/{_cache_safe_name(name)}_{nsamples}_{seqlen}_{seed}_{batch_size}.pt"
    )
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        return traindataset
    if _is_mixture_dataset(name):
        traindataset = _build_mixture_calibration_data(
            name=name,
            tokenizer=tokenizer,
            nsamples=nsamples,
            seqlen=seqlen,
            seed=seed,
            batch_size=batch_size,
            dataset_cache_dir=dataset_cache_dir,
        )
    else:
        traindataset = _build_calibration_chunks_from_texts(
            texts=_load_training_texts(name, dataset_cache_dir),
            tokenizer=tokenizer,
            nsamples=nsamples,
            seqlen=seqlen,
            seed=seed,
            batch_size=batch_size,
        )
    torch.save(traindataset, cache_file)
    return traindataset


def _sample_from_joined_training_texts(name, nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    return _sample_loader_from_texts(
        texts=_load_training_texts(name, dataset_cache_dir),
        nsamples=nsamples,
        seed=seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
    )



def get_wikitext2(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=dataset_cache_dir)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=dataset_cache_dir)

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', cache_dir=dataset_cache_dir)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
    valdata = load_dataset("json", data_files="utils/c4-validation.json")['train']

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 



def get_ptb_new(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test', cache_dir=dataset_cache_dir)

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
    valdata = load_dataset("json", data_files="utils/c4-validation.json")['train']

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, tokenizer)
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, tokenizer)
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if _is_mixture_dataset(name):
        return _build_mixture_loader(name, nsamples, seed, seqlen, tokenizer), None
    if _is_evol_codealpaca(name):
        return _sample_from_joined_training_texts(name, nsamples, seed, seqlen, tokenizer), None
    if _is_tulu_math(name):
        return _sample_from_joined_training_texts(name, nsamples, seed, seqlen, tokenizer), None
    
    
    
def get_test_data(name, tokenizer, seq_len=2048, batch_size = 4):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)
    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    ####
    if 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        test_data = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
