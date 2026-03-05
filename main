"""
Sentiment Triage Assistant (PyTorch)
====================================
A small, practical NLP assistant that:
1) Trains a sentiment classifier from scratch (no pretrained models).
2) Predicts sentiment for user text in an interactive loop.
3) Produces a purpose-driven action recommendation (triage guidance).
4) Accepts user feedback labels during chat and can quickly fine-tune.
5) Can save/load model checkpoints for reuse.

Architecture (as requested)
---------------------------
- Lowercase whitespace tokenizer
- Vocabulary with <PAD> and <UNK>
- Embedding(dim=32)
- Mean pooling over sequence
- Linear(32 -> 16) + ReLU
- Dropout(0.2)
- Linear(16 -> 1)
- BCEWithLogitsLoss for training
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1) Reproducibility + configuration
# ============================================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("sentiment_triage_model.pt")


@dataclass
class Config:
    embedding_dim: int = 32
    hidden_dim: int = 16
    dropout: float = 0.2
    lr: float = 0.001
    batch_size: int = 8
    epochs: int = 35
    early_stopping_patience: int = 6


CFG = Config()


# ============================================================
# 2) Custom balanced dataset
#    label: 1 = positive, 0 = negative
# ============================================================
RAW_DATA: List[Tuple[str, int]] = [
    # Positive (40)
    ("i absolutely loved this movie", 1),
    ("the food was amazing and fresh", 1),
    ("what a fantastic experience", 1),
    ("i am very happy with the service", 1),
    ("this product works perfectly", 1),
    ("the design is beautiful and clean", 1),
    ("i enjoyed every minute", 1),
    ("super friendly staff and quick help", 1),
    ("the package arrived early", 1),
    ("quality is excellent for the price", 1),
    ("i would definitely buy this again", 1),
    ("great value and easy to use", 1),
    ("the app is smooth and intuitive", 1),
    ("everything turned out better than expected", 1),
    ("this was a pleasant surprise", 1),
    ("i feel satisfied and impressed", 1),
    ("performance is fast and reliable", 1),
    ("the tutorial was clear and helpful", 1),
    ("my order was correct and complete", 1),
    ("the camera quality is outstanding", 1),
    ("customer support solved my issue quickly", 1),
    ("i had a wonderful time", 1),
    ("the update made everything better", 1),
    ("this is one of my favorite purchases", 1),
    ("simple setup and great results", 1),
    ("excellent build quality and awesome battery", 1),
    ("the lesson was engaging and useful", 1),
    ("service was polite and professional", 1),
    ("this feels premium and dependable", 1),
    ("the interface is clean and responsive", 1),
    ("support gave me a clear solution", 1),
    ("refund process was smooth and fair", 1),
    ("delivery was quick and on time", 1),
    ("the team listened and fixed the issue", 1),
    ("setup instructions were easy to follow", 1),
    ("i am delighted with the final result", 1),
    ("this made my day better", 1),
    ("great communication and fast response", 1),
    ("the product exceeded my expectations", 1),
    ("very pleased with this purchase", 1),
    # Negative (40)
    ("i hated this movie", 0),
    ("the food was cold and tasteless", 0),
    ("what a terrible experience", 0),
    ("i am very disappointed with the service", 0),
    ("this product stopped working", 0),
    ("the design looks cheap and ugly", 0),
    ("i regret buying this", 0),
    ("rude staff and no support", 0),
    ("the package arrived damaged", 0),
    ("quality is poor for the price", 0),
    ("i would never buy this again", 0),
    ("bad value and hard to use", 0),
    ("the app is slow and confusing", 0),
    ("everything went worse than expected", 0),
    ("this was a complete waste", 0),
    ("i feel frustrated and annoyed", 0),
    ("performance is unstable and laggy", 0),
    ("the tutorial was unclear and useless", 0),
    ("my order was wrong and incomplete", 0),
    ("the camera quality is terrible", 0),
    ("customer support ignored my issue", 0),
    ("i had an awful time", 0),
    ("the update broke important features", 0),
    ("this is one of my worst purchases", 0),
    ("setup was difficult and results were bad", 0),
    ("terrible build quality and weak battery", 0),
    ("the lesson was boring and confusing", 0),
    ("service was rude and unhelpful", 0),
    ("this feels fragile and unreliable", 0),
    ("the interface is cluttered and buggy", 0),
    ("support never responded to my ticket", 0),
    ("refund process was slow and painful", 0),
    ("delivery was late and inaccurate", 0),
    ("the team ignored my concerns", 0),
    ("instructions were incomplete and confusing", 0),
    ("i am upset with the final result", 0),
    ("this ruined my day", 0),
    ("communication was poor and delayed", 0),
    ("the product failed after one day", 0),
    ("very disappointed with this purchase", 0),
]


# ============================================================
# 3) Tokenizer, vocabulary, encoding, padding
# ============================================================
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def tokenize(text: str) -> List[str]:
    """Lowercase + basic whitespace tokenization."""
    return text.lower().split()


def build_vocab(data: Sequence[Tuple[str, int]]) -> Dict[str, int]:
    """
    Build token -> ID mapping from provided dataset.

    Includes special tokens:
      - <PAD> at index 0 (for sequence padding)
      - <UNK> at index 1 (for unseen words)
    """
    counter = Counter()
    for text, _ in data:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for token in counter:
        vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int]) -> List[int]:
    """Convert input text into token IDs, using <UNK> for out-of-vocab tokens."""
    return [vocab.get(token, UNK_IDX) for token in tokenize(text)]


def pad_right(token_ids: List[int], max_len: int) -> List[int]:
    """Right-pad sequence to max_len with PAD_IDX."""
    return token_ids + [PAD_IDX] * (max_len - len(token_ids))


# ============================================================
# 4) Train/val/test split (70/10/20)
# ============================================================
def split_data(samples: List[Tuple[List[int], int]]) -> Tuple[List[Tuple[List[int], int]], List[Tuple[List[int], int]], List[Tuple[List[int], int]]]:
    random.shuffle(samples)
    n_total = len(samples)
    n_train = int(0.7 * n_total)
    n_val = int(0.1 * n_total)

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]
    return train, val, test


# ============================================================
# 5) Dataset + DataLoader
# ============================================================
class SentimentDataset(Dataset):
    """
    Each item:
      x: LongTensor [max_len]  (token IDs)
      y: FloatTensor scalar    (0.0 or 1.0)
    """

    def __init__(self, samples: Sequence[Tuple[List[int], int]], max_len: int):
        self.samples = list(samples)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, label = self.samples[idx]
        x = torch.tensor(pad_right(seq, self.max_len), dtype=torch.long)
        y = torch.tensor(float(label), dtype=torch.float32)
        return x, y


# ============================================================
# 6) Model architecture
# ============================================================
class SentimentNet(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tensor shapes:
        - Input x: [batch_size, seq_len]
        - Embeddings: [batch_size, seq_len, 32]

        Forward pass:
        1) Embedding lookup
        2) Masked mean pooling over sequence length (ignore PAD tokens)
        3) Feedforward layers
        4) Return logits [batch_size]

        NOTE: Sigmoid is NOT applied here because BCEWithLogitsLoss expects logits.
        """
        emb = self.embedding(x)

        # Mask PAD tokens so they do not contribute to pooled representation.
        mask = (x != PAD_IDX).unsqueeze(-1).float()    # [B, T, 1]
        emb = emb * mask                                # [B, T, E]

        token_counts = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        pooled = emb.sum(dim=1) / token_counts         # [B, E]

        h = self.relu(self.fc1(pooled))                 # [B, hidden]
        h = self.dropout(h)                             # [B, hidden]
        logits = self.fc2(h).squeeze(1)                 # [B]
        return logits


# ============================================================
# 7) Training / evaluation helpers
# ============================================================
def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == labels).float().mean().item()


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy_from_logits(logits, y_batch)
            batches += 1

    if batches == 0:
        return 0.0, 0.0
    return total_loss / batches, total_acc / batches


def predict_probability(model: nn.Module, text: str, vocab: Dict[str, int], max_len: int, device: torch.device) -> float:
    token_ids = encode(text, vocab)
    padded = pad_right(token_ids, max_len)
    x = torch.tensor([padded], dtype=torch.long, device=device)  # [1, max_len]

    model.eval()
    with torch.no_grad():
        logit = model(x)
        prob_positive = torch.sigmoid(logit).item()
    return prob_positive


# ============================================================
# 8) Practical triage logic (genuine purpose)
# ============================================================

POSITIVE_HINTS = {"love", "great", "excellent", "amazing", "happy", "helpful", "smooth", "wonderful", "outstanding"}
NEGATIVE_HINTS = {"worst", "awful", "terrible", "hate", "hated", "broken", "refund", "bad", "disappointed", "frustrated", "annoyed"}


def hybrid_probability(model: nn.Module, text: str, vocab: Dict[str, int], max_len: int, device: torch.device) -> float:
    """Blend neural score with light lexical priors for better small-data robustness."""
    nn_prob = predict_probability(model, text, vocab, max_len, device)
    tokens = set(tokenize(text))
    pos_hits = len(tokens.intersection(POSITIVE_HINTS))
    neg_hits = len(tokens.intersection(NEGATIVE_HINTS))
    delta = pos_hits - neg_hits
    lex_adjustment = max(-0.4, min(0.4, 0.15 * delta))
    lex_prior = 0.5 + lex_adjustment

    # Heavier lexical contribution helps stability on tiny custom datasets.
    blended = 0.55 * nn_prob + 0.45 * lex_prior
    return max(0.0, min(1.0, blended))
HIGH_PRIORITY_TERMS = {
    "angry", "refund", "broken", "crash", "failed", "fraud", "cancel", "lawsuit",
    "horrible", "worst", "unacceptable", "scam", "urgent", "immediately",
}


def classify_urgency(text: str, prob_positive: float) -> str:
    """
    Heuristic urgency for operations/customer-support triage.
    - High urgency if sentiment is strongly negative or contains escalation terms.
    - Medium urgency for somewhat negative feedback.
    - Low urgency otherwise.
    """
    tokens = set(tokenize(text))
    has_priority_term = len(tokens.intersection(HIGH_PRIORITY_TERMS)) > 0

    if prob_positive < 0.25 or has_priority_term:
        return "high"
    if prob_positive < 0.45:
        return "medium"
    return "low"


def triage_recommendation(prob_positive: float, urgency: str) -> str:
    if urgency == "high":
        return "Escalate to human support immediately and acknowledge the issue with priority."
    if urgency == "medium":
        return "Open a support ticket, request details, and follow up within one business day."
    if prob_positive >= 0.7:
        return "Ask for testimonial/review and suggest related product tips."
    return "Send a polite response and offer optional assistance."


# ============================================================
# 9) Build data pipeline, train with early stopping, evaluate
# ============================================================
def prepare_dataloaders(data: Sequence[Tuple[str, int]], batch_size: int) -> Tuple[Dict[str, int], int, DataLoader, DataLoader, DataLoader, List[Tuple[List[int], int]]]:
    vocab = build_vocab(data)
    encoded = [(encode(text, vocab), label) for text, label in data]
    max_len = max(len(seq) for seq, _ in encoded)

    train_samples, val_samples, test_samples = split_data(encoded)

    train_loader = DataLoader(SentimentDataset(train_samples, max_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SentimentDataset(val_samples, max_len), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SentimentDataset(test_samples, max_len), batch_size=batch_size, shuffle=False)

    return vocab, max_len, train_loader, val_loader, test_loader, train_samples


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: Config, device: torch.device) -> Tuple[nn.Module, float]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_loss = float("inf")
    patience_left = cfg.early_stopping_patience
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = cfg.early_stopping_patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


def evaluate_test(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.BCEWithLogitsLoss()
    return run_epoch(model, test_loader, criterion, None, device)


# ============================================================
# 10) Checkpointing
# ============================================================
def save_checkpoint(model: nn.Module, vocab: Dict[str, int], max_len: int, path: Path) -> None:
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "max_len": max_len}, path)


def load_checkpoint(path: Path, cfg: Config, device: torch.device) -> Tuple[SentimentNet, Dict[str, int], int]:
    ckpt = torch.load(path, map_location=device)
    vocab = ckpt["vocab"]
    max_len = ckpt["max_len"]

    model = SentimentNet(
        vocab_size=len(vocab),
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, vocab, max_len


# ============================================================
# 11) Main script flow
# ============================================================
def main() -> None:
    print("Preparing data and training sentiment model...")
    vocab, max_len, train_loader, val_loader, test_loader, train_samples = prepare_dataloaders(RAW_DATA, CFG.batch_size)

    model = SentimentNet(
        vocab_size=len(vocab),
        embedding_dim=CFG.embedding_dim,
        hidden_dim=CFG.hidden_dim,
        dropout=CFG.dropout,
    ).to(DEVICE)

    model, best_val_loss = train_model(model, train_loader, val_loader, CFG, DEVICE)
    test_loss, test_acc = evaluate_test(model, test_loader, DEVICE)

    print(f"\nBest Validation Loss: {best_val_loss:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # Save trained model so it can be reused quickly.
    save_checkpoint(model, vocab, max_len, MODEL_PATH)
    print(f"Saved checkpoint to: {MODEL_PATH}")

    # Show fixed examples first.
    examples = [
        "i am happy with the quick support",
        "this app is broken and i want a refund now",
        "it works fine but setup was a bit slow",
        "excellent service and smooth experience",
        "worst purchase i made this year",
    ]

    print("\nExample predictions:")
    for text in examples:
        prob_pos = hybrid_probability(model, text, vocab, max_len, DEVICE)
        sentiment = "positive" if prob_pos >= 0.5 else "negative"
        urgency = classify_urgency(text, prob_pos)
        advice = triage_recommendation(prob_pos, urgency)
        print(f'- "{text}" -> sentiment={sentiment}, positive_prob={prob_pos:.4f}, urgency={urgency}')
        print(f"  recommendation: {advice}")

    # Interactive loop with useful commands.
    print("\n--- Sentiment Triage Assistant ---")
    print("Type any sentence to classify sentiment and receive triage advice.")
    print("Commands:")
    print("  /save                -> save current model checkpoint")
    print("  /load                -> load checkpoint from disk")
    print("  /feedback <0|1> <text> -> add labeled example and quick fine-tune")
    print("  quit                 -> exit")

    # Keep local mutable training pool for feedback-driven updates.
    feedback_pool = list(train_samples)

    while True:
        user_in = input("You: ").strip()
        if user_in.lower() in {"quit", "exit", "q"}:
            print("Bot: Goodbye!")
            break

        if not user_in:
            print("Bot: Please enter non-empty text.")
            continue

        if user_in == "/save":
            save_checkpoint(model, vocab, max_len, MODEL_PATH)
            print(f"Bot: Model saved to {MODEL_PATH}")
            continue

        if user_in == "/load":
            if MODEL_PATH.exists():
                model_loaded, vocab_loaded, max_len_loaded = load_checkpoint(MODEL_PATH, CFG, DEVICE)
                model = model_loaded
                vocab = vocab_loaded
                max_len = max_len_loaded
                print(f"Bot: Loaded model from {MODEL_PATH}")
            else:
                print("Bot: No checkpoint found yet.")
            continue

        if user_in.startswith("/feedback "):
            # Format: /feedback 0 this was awful
            #      or /feedback 1 this is great
            parts = user_in.split(maxsplit=2)
            if len(parts) < 3 or parts[1] not in {"0", "1"}:
                print("Bot: Usage -> /feedback <0|1> <text>")
                continue

            label = int(parts[1])
            text = parts[2]
            feedback_pool.append((encode(text, vocab), label))

            # Quick fine-tune step on accumulated feedback-enhanced pool.
            ft_loader = DataLoader(SentimentDataset(feedback_pool, max_len), batch_size=CFG.batch_size, shuffle=True)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

            for _ in range(2):
                run_epoch(model, ft_loader, criterion, optimizer, DEVICE)

            print("Bot: Thanks, feedback applied with quick fine-tuning.")
            continue

        prob_pos = hybrid_probability(model, user_in, vocab, max_len, DEVICE)
        sentiment = "positive" if prob_pos >= 0.5 else "negative"
        urgency = classify_urgency(user_in, prob_pos)
        advice = triage_recommendation(prob_pos, urgency)

        print(f"Bot: sentiment={sentiment} | positive_prob={prob_pos:.4f} | urgency={urgency}")
        print(f"Bot: recommendation: {advice}")


if __name__ == "__main__":
    main()


"""
Potential next improvements
---------------------------
1) Replace mean pooling with LSTM/GRU/attention for better sequence understanding.
2) Use a much larger domain-specific dataset and stratified splitting.
3) Add confidence calibration and threshold tuning on validation data.
4) Track precision/recall/F1 and confusion matrix.
5) Build a real voice interface with speech-to-text and text-to-speech.
6) For genuine LLM behavior, switch to a generative architecture and pretraining.
"""
