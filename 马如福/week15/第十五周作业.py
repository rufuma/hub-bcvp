DEFAULT_VOCAB_SIZE = 300
START_IDX = 256
BASE_VOCAB_SIZE = 256


def read_file_context(file_path, encoding='utf-8'):
    file_context = None
    try:
        with open(file_path, encoding=encoding) as f:
            file_context = f.read()
        if not file_context:
            print("警告！文件内容为空！")
    except Exception as e:
        print(e)
    finally:
        return file_context

def text_to_tokens(context, tokens="utf-8"):
    if not context:
        return []
    return list(map(int, context.encode(tokens)))

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
      if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
        newids.append(idx)
        i += 2
      else:
        newids.append(ids[i])
        i += 1
    return newids


def train_bpe_merges(tokens):
    vocab_size = DEFAULT_VOCAB_SIZE
    num_merges = vocab_size - BASE_VOCAB_SIZE
    ids = list(tokens)

    merges = {} # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = START_IDX + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return merges, ids

def encode(text, merges):
    # given a string, return list of integers (the tokens)
    tokens = text_to_tokens(text)
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
          break # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

def decode(ids, merges):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    # given ids (list of integers), return Python string
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


if __name__ == "__main__":
    context = read_file_context("bpe_text.txt")
    tokens = text_to_tokens(context)
    merges, ids = train_bpe_merges(tokens)
    encoder = encode(context, merges)
    decoder = decode(encoder, merges)
    print("tokens length:", len(tokens))
    print("ids length:", len(ids))
    print(f"compression ratio: {len(tokens) / len(ids):.2f}X")
    if context == decoder:
        print("转码成功")
    else:
        print("转码失败")
