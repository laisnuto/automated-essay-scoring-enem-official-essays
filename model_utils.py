import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
    get_linear_schedule_with_warmup
)
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

SCORES = [0, 40, 80, 120, 160, 200]
score_to_class = {s: i for i, s in enumerate(SCORES)}
class_to_score = {i: s for i, s in enumerate(SCORES)}

COMPETENCIES = [1, 2, 3, 4, 5]
MODEL_TEMPLATES = {}
TOKENIZER_NAMES = {
    "bertimbau": "neuralmind/bert-base-portuguese-cased"
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def round_to_nearest_40(x: int) -> int:
    """Convert score (multiples of 20) to nearest class (multiples of 40). Used only for training."""
    x = int(x)
    return SCORES[int(np.argmin([abs(x - s) for s in SCORES]))]


def to_class(y_score: int) -> int:
    return score_to_class[round_to_nearest_40(y_score)]


class EnemCompDataset(Dataset):
    def __init__(
        self,
        df,
        comp_col: str,
        tokenizer,
        for_train: bool = True,
        max_len: int = 512,
        text_col: str = "texto"
    ):
        self.texts = df[text_col].astype(str).tolist()
        self.tokenizer = tokenizer
        self.for_train = for_train
        self.max_len = max_len
        
        if for_train:
            self.labels = [to_class(v) for v in df[comp_col].tolist()]
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict:
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        
        if self.for_train:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


def load_tokenizer_and_model(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    num_labels: int = 6,
    device: Optional[torch.device] = None,
    max_len: int = 512
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    if device is None:
        device = get_device()
    
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if not hasattr(tokenizer, 'model_max_length') or tokenizer.model_max_length is None:
        tokenizer.model_max_length = max_len
    elif tokenizer.model_max_length > max_len:
        tokenizer.model_max_length = max_len
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    ).to(device)
    
    return tokenizer, model


def create_pipeline(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    device: Optional[torch.device] = None,
    max_len: int = 512
) -> TextClassificationPipeline:
    if device is None:
        device = get_device()
    
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if not hasattr(tokenizer, 'model_max_length') or tokenizer.model_max_length is None:
            tokenizer.model_max_length = max_len
        elif tokenizer.model_max_length > max_len:
            tokenizer.model_max_length = max_len
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model = model.to(device)
    
    device_id = 0 if torch.cuda.is_available() else -1
    
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_all_scores=False,
        device=device_id,
        truncation=True,
        max_length=max_len
    )
    
    return pipe


def train_model_cv(
    df_train,
    comp_idx: int,
    hyperparams: Dict,
    model_name_template: str,
    tokenizer_name: Optional[str] = None,
    cv_folds: Optional[int] = None,
    text_col: str = "texto",
    year_col: str = "ano",
    max_len: int = 512,
    device: Optional[torch.device] = None
) -> float:
    """
    Train model with cross-validation for hyperparameter search.
    Each fold is a different year (leave-one-year-out validation).
    Returns average QWK across folds.
    """
    if device is None:
        device = get_device()
    
    comp_col = f"C{comp_idx}"
    model_name = model_name_template.format(comp_idx)
    
    tokenizer, model = load_tokenizer_and_model(
        model_name,
        tokenizer_name=tokenizer_name,
        num_labels=6,
        device=device,
        max_len=max_len
    )
    
    unique_years = sorted(df_train[year_col].unique())
    
    if cv_folds is None:
        cv_folds = len(unique_years)
    
    from sklearn.metrics import cohen_kappa_score
    qwk_scores = []
    
    for fold, val_year in enumerate(unique_years[:cv_folds]):
        df_fold_train = df_train[df_train[year_col] != val_year].reset_index(drop=True)
        df_fold_val = df_train[df_train[year_col] == val_year].reset_index(drop=True)
        
        if len(df_fold_val) < 2:
            continue
        
        train_ds_fold = EnemCompDataset(
            df_fold_train, comp_col, tokenizer, for_train=True, max_len=max_len, text_col=text_col
        )
        val_ds_fold = EnemCompDataset(
            df_fold_val, comp_col, tokenizer, for_train=True, max_len=max_len, text_col=text_col
        )
        
        train_loader = DataLoader(
            train_ds_fold, batch_size=hyperparams['batch_size'], shuffle=True
        )
        val_loader = DataLoader(
            val_ds_fold, batch_size=hyperparams['batch_size'], shuffle=False
        )
        
        model_fold = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=6
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model_fold.parameters(), lr=hyperparams['learning_rate']
        )
        
        model_fold.train()
        for epoch in range(hyperparams['epochs']):
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                
                outputs = model_fold(**batch)
                loss = criterion(outputs.logits, batch["labels"])
                loss.backward()
                optimizer.step()
        
        model_fold.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model_fold(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch["labels"].cpu().numpy())
        
        val_preds_scores = [class_to_score[p] for p in val_preds]
        val_labels_scores = [class_to_score[l] for l in val_labels]
        
        ALL_LABELS = list(range(0, 201, 20))
        qwk = cohen_kappa_score(
            val_labels_scores, val_preds_scores,
            weights='quadratic', labels=ALL_LABELS
        )
        qwk_scores.append(qwk)
    
    if not qwk_scores:
        return 0.0
    
    return np.mean(qwk_scores)


def train_final_model(
    df_train,
    comp_idx: int,
    hyperparams: Dict,
    model_name_template: str,
    tokenizer_name: Optional[str] = None,
    save_path: Optional[str] = None,
    text_col: str = "texto",
    max_len: int = 512,
    device: Optional[torch.device] = None
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, str]:
    if device is None:
        device = get_device()
    
    comp_col = f"C{comp_idx}"
    model_name = model_name_template.format(comp_idx)
    
    tokenizer, model = load_tokenizer_and_model(
        model_name,
        tokenizer_name=tokenizer_name,
        num_labels=6,
        device=device,
        max_len=max_len
    )
    
    train_ds = EnemCompDataset(
        df_train, comp_col, tokenizer, for_train=True, max_len=max_len, text_col=text_col
    )
    
    loader_kwargs = dict(
        batch_size=hyperparams['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )
    train_loader = DataLoader(train_ds, **loader_kwargs)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'])
    total_steps = max(1, hyperparams['epochs'] * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps
    )
    
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    from tqdm.auto import tqdm
    import time
    
    for ep in range(1, hyperparams['epochs'] + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        
        for batch in tqdm(
            train_loader,
            desc=f"[C{comp_idx}] Epoch {ep}/{hyperparams['epochs']} (final)",
            leave=False
        ):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=(device.type == "cuda")):
                logits = model(**batch).logits
                loss = criterion(logits, batch["labels"])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running += loss.item()
        
        train_loss = running / max(1, len(train_loader))
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    
    return tokenizer, model, save_path


def predict_scores(
    df_split,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    comp_idx: int,
    text_col: str = "texto",
    max_len: int = 512,
    batch_size: int = 16,
    device: Optional[torch.device] = None
) -> List[int]:
    if device is None:
        device = get_device()
    
    comp_col = f"C{comp_idx}"
    ds = EnemCompDataset(
        df_split, comp_col, tokenizer=tokenizer, for_train=False,
        max_len=max_len, text_col=text_col
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    model.eval()
    preds_cls = []
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            preds_cls.extend(preds)
    
    return [class_to_score[c] for c in preds_cls]


def calculate_expected_grade(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    device: Optional[torch.device] = None,
    max_len: int = 512
) -> float:
    """Calculate expected grade using softmax and weighted sum of class probabilities."""
    if device is None:
        device = get_device()
    
    import torch.nn.functional as F
    
    inputs = tokenizer(
        text, return_tensors="pt", padding=True,
        truncation=True, max_length=max_len
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = F.softmax(outputs.logits, dim=-1)[0]
    
    expected_grade = (
        0 * probabilities[0] +
        40 * probabilities[1] +
        80 * probabilities[2] +
        120 * probabilities[3] +
        160 * probabilities[4] +
        200 * probabilities[5]
    ).item()
    
    return expected_grade
