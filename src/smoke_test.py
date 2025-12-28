# Smoke test: GPU + sentence-transformers + cosine sim + topk
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

print("numpy:", np.__version__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available"

model_name = "sentence-transformers/all-MiniLM-L6-v2"
m = SentenceTransformer(model_name, device="cuda")
print("model:", model_name, "dim:", m.get_sentence_embedding_dimension())

texts = ["refund policy for my purchase", "how to reset my password", "what are the benefits of the program"]
emb = m.encode(texts, convert_to_tensor=True, normalize_embeddings=True)  # (3, 384) on GPU
print("emb device:", emb.device, "shape:", tuple(emb.shape))

# Cosine similarity (since normalized, dot product == cosine)
S = emb @ emb.T
print("sim matrix:\n", S.detach().cpu().numpy().round(3))

# top-2 neighbors for each row
vals, idx = torch.topk(S, k=2, dim=1)
print("top2 idx:", idx.detach().cpu().tolist())
print("top2 vals:", vals.detach().cpu().tolist())

print("SMOKE TEST: OK")
