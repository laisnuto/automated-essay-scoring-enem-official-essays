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
import pandas as pd
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
    model_name_template: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    cv_folds: Optional[int] = None,
    text_col: str = "texto",
    year_col: str = "ano",
    max_len: int = 512,
    device: Optional[torch.device] = None,
    base_model_name: Optional[str] = None,
    gradient_clipping: bool = False
) -> float:
    """
    Train model with cross-validation for hyperparameter search.
    Each fold is a different year (leave-one-year-out validation).
    Returns average QWK across folds.
    """
    if device is None:
        device = get_device()
    
    comp_col = f"C{comp_idx}"
    
    # Load tokenizer and model
    if base_model_name:
        # For original models, use base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length is None:
            tokenizer.model_max_length = max_len
        elif hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length > max_len:
            tokenizer.model_max_length = max_len
        model_name = base_model_name
    else:
        # For fine-tuned models, use template
        model_name = model_name_template.format(comp_idx)
        if tokenizer_name:
            # Use separate tokenizer (e.g., BERTimbau)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length is None:
                tokenizer.model_max_length = max_len
            elif hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length > max_len:
                tokenizer.model_max_length = max_len
        else:
            # Use model's tokenizer
            tokenizer, _ = load_tokenizer_and_model(
                model_name,
                tokenizer_name=None,
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
                
                # Apply gradient clipping if requested
                if gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model_fold.parameters(), max_norm=1.0)
                
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




def run_grid_search_for_competency(
    df_train,
    comp_idx: int,
    hyperparameter_space: Dict[str, List],
    model_name_template: str,
    tokenizer_name: Optional[str] = None,
    year_col: str = "ano",
    text_col: str = "texto",
    max_len: int = 512,
    cv_folds: Optional[int] = None,
    device: Optional[torch.device] = None,
    checkpoint_file: Optional[str] = None,
    best_hyperparams: Optional[Dict] = None,
    best_qwk_scores: Optional[Dict] = None,
    use_base_model: bool = False,
    base_model_name: Optional[str] = None,
    gradient_clipping: bool = False
) -> Tuple[Dict, float]:
    """
    Run grid search for a single competency.
    
    Args:
        df_train: Training dataframe
        comp_idx: Competency index (1-5)
        hyperparameter_space: Dict with lists of values for each hyperparameter
        model_name_template: Template string for model name (e.g., "model-C{}")
        tokenizer_name: Optional tokenizer name (if different from model)
        year_col: Column name for year
        text_col: Column name for text
        max_len: Maximum sequence length
        cv_folds: Number of CV folds (None = use all years)
        device: PyTorch device
        checkpoint_file: Path to checkpoint file for saving progress
        best_hyperparams: Dict to store best hyperparameters (will be updated)
        best_qwk_scores: Dict to store best QWK scores (will be updated)
        use_base_model: If True, use base_model_name instead of model_name_template
        base_model_name: Base model name when use_base_model=True
        gradient_clipping: Whether to apply gradient clipping
    
    Returns:
        Tuple of (best_hyperparams_dict, best_qwk_score)
    """
    import itertools
    import json
    
    if device is None:
        device = get_device()
    
    if best_hyperparams is None:
        best_hyperparams = {}
    if best_qwk_scores is None:
        best_qwk_scores = {}
    
    comp_key = f'C{comp_idx}'
    
    # Check if already completed
    if comp_key in best_hyperparams:
        print(f"{comp_key} already completed. Skipping.")
        print(f"  Best params: {best_hyperparams[comp_key]}")
        print(f"  Best QWK: {best_qwk_scores.get(comp_key, 0):.3f}")
        return best_hyperparams[comp_key], best_qwk_scores.get(comp_key, 0.0)
    
    print(f"\n=== Searching hyperparameters for {comp_key} ===")
    
    best_qwk = -1
    best_params = None
    
    # Generate all combinations
    param_combinations = list(itertools.product(*hyperparameter_space.values()))
    param_names = list(hyperparameter_space.keys())
    
    for i, params in enumerate(param_combinations):
        hyperparams = dict(zip(param_names, params))
        print(f"\nTrying combination {i+1}/{len(param_combinations)}: {hyperparams}")
        
        try:
            if use_base_model and base_model_name:
                # For original models, use base model
                qwk = train_model_cv(
                    df_train, comp_idx, hyperparams,
                    model_name_template=None,
                    tokenizer_name=None,
                    base_model_name=base_model_name,
                    year_col=year_col,
                    text_col=text_col,
                    max_len=max_len,
                    cv_folds=cv_folds,
                    device=device,
                    gradient_clipping=gradient_clipping
                )
            else:
                qwk = train_model_cv(
                    df_train, comp_idx, hyperparams,
                    model_name_template=model_name_template,
                    tokenizer_name=tokenizer_name,
                    year_col=year_col,
                    text_col=text_col,
                    max_len=max_len,
                    cv_folds=cv_folds,
                    device=device,
                    gradient_clipping=False
                )
            
            if qwk > best_qwk:
                best_qwk = qwk
                best_params = hyperparams.copy()
        
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    best_hyperparams[comp_key] = best_params
    best_qwk_scores[comp_key] = best_qwk
    
    print(f"\nBest hyperparameters for {comp_key}:")
    print(f"  Params: {best_params}")
    print(f"  QWK: {best_qwk:.3f}")
    
    # Save checkpoint if provided
    if checkpoint_file:
        checkpoint_data = {
            'best_hyperparams': best_hyperparams,
            'best_qwk_scores': best_qwk_scores
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"✓ Checkpoint saved to {checkpoint_file}")
    
    return best_params, best_qwk


def run_grid_search_all_competencies(
    df_train,
    hyperparameter_space: Dict[str, List],
    model_name_template: str,
    tokenizer_name: Optional[str] = None,
    competencies: List[int] = [1, 2, 3, 4, 5],
    year_col: str = "ano",
    text_col: str = "texto",
    max_len: int = 512,
    cv_folds: Optional[int] = None,
    device: Optional[torch.device] = None,
    checkpoint_file: Optional[str] = None,
    use_base_model: bool = False,
    base_model_name: Optional[str] = None,
    gradient_clipping: bool = False
) -> Tuple[Dict, Dict]:
    """
    Run grid search for all competencies.
    
    Returns:
        Tuple of (best_hyperparams_dict, best_qwk_scores_dict)
    """
    import json
    
    if device is None:
        device = get_device()
    
    # Load checkpoint if exists
    best_hyperparams = {}
    best_qwk_scores = {}
    
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                best_hyperparams = checkpoint_data.get('best_hyperparams', {})
                best_qwk_scores = checkpoint_data.get('best_qwk_scores', {})
                print(f"Loaded checkpoint from {checkpoint_file}")
                print(f"Found saved results for: {list(best_hyperparams.keys())}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            best_hyperparams = {}
            best_qwk_scores = {}
    
    # Run grid search for each competency
    for comp_idx in competencies:
        run_grid_search_for_competency(
            df_train, comp_idx, hyperparameter_space,
            model_name_template, tokenizer_name,
            year_col=year_col, text_col=text_col, max_len=max_len,
            cv_folds=cv_folds, device=device,
            checkpoint_file=checkpoint_file,
            best_hyperparams=best_hyperparams,
            best_qwk_scores=best_qwk_scores,
            use_base_model=use_base_model,
            base_model_name=base_model_name,
            gradient_clipping=gradient_clipping
        )
    
    # Save final results
    if checkpoint_file:
        checkpoint_data = {
            'best_hyperparams': best_hyperparams,
            'best_qwk_scores': best_qwk_scores
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    return best_hyperparams, best_qwk_scores


def train_final_models_all_competencies(
    df_train,
    best_hyperparams: Dict,
    model_name_template: str,
    tokenizer_name: Optional[str] = None,
    competencies: List[int] = [1, 2, 3, 4, 5],
    save_dir: Optional[str] = None,
    model_name_prefix: str = "model",
    year_col: str = "ano",
    text_col: str = "texto",
    max_len: int = 512,
    device: Optional[torch.device] = None,
    use_base_model: bool = False,
    base_model_name: Optional[str] = None
) -> Tuple[Dict[int, AutoTokenizer], Dict[int, AutoModelForSequenceClassification]]:
    """
    Train final models for all competencies using best hyperparameters.
    
    Returns:
        Tuple of (tokenizers_dict, models_dict)
    """
    if device is None:
        device = get_device()
    
    tokenizers_final = {}
    models_final = {}
    
    for comp_idx in competencies:
        comp_key = f'C{comp_idx}'
        hyperparams = best_hyperparams.get(
            comp_key,
            {'learning_rate': 2e-5, 'batch_size': 16, 'epochs': 5}
        )
        
        print(f"\n=== Training Final Model — Competência {comp_key} ===")
        
        if save_dir:
            save_path = os.path.join(save_dir, f"{model_name_prefix}_C{comp_idx}_finetuned_com_redacoes_oficiais")
        else:
            save_path = None
        
        if use_base_model and base_model_name:
            # For original models
            comp_col = f"C{comp_idx}"
            tokenizer, model = load_tokenizer_and_model(
                base_model_name,
                tokenizer_name=base_model_name,
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
                    desc=f"[{comp_key}] Epoch {ep}/{hyperparams['epochs']} (final)",
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
                print(f"[{comp_key}] epoch {ep}/{hyperparams['epochs']} - train loss: {train_loss:.4f} | tempo: {time.time()-t0:.1f}s")
                
                if save_path:
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
            
            if save_path:
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"[{comp_key}] ✓ Modelo final salvo em: {save_path}")
        else:
            # For fine-tuned models
            tok, mdl, _ = train_final_model(
                df_train, comp_idx, hyperparams,
                model_name_template=model_name_template,
                tokenizer_name=tokenizer_name,
                save_path=save_path,
                text_col=text_col,
                max_len=max_len,
                device=device
            )
            tokenizer, model = tok, mdl
        
        tokenizers_final[comp_idx] = tokenizer
        models_final[comp_idx] = model
    
    return tokenizers_final, models_final


def evaluate_final_models(
    df_test,
    tokenizers_final: Dict[int, AutoTokenizer],
    models_final: Dict[int, AutoModelForSequenceClassification],
    competencies: List[int] = [1, 2, 3, 4, 5],
    text_col: str = "texto",
    max_len: int = 512,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    save_csv_path: Optional[str] = None,
    year_col: str = "ano"
) -> pd.DataFrame:
    """
    Evaluate final models on test set and return predictions dataframe.
    
    Returns:
        DataFrame with predictions and ground truth
    """
    if device is None:
        device = get_device()
    
    # Prepare test results
    df_test_final = df_test.reset_index(drop=True).copy()
    df_test_final["id"] = df_test_final.index
    
    # Base CSV structure
    cols_base = {"id": df_test_final["id"].values}
    if year_col in df_test_final.columns:
        cols_base[year_col] = df_test_final[year_col].values
    out_final = pd.DataFrame(cols_base)
    
    # Ground truth scores
    for c in competencies:
        out_final[f"C{c}"] = pd.to_numeric(df_test_final[f"C{c}"], errors="coerce").astype("Int64")
    
    # Predictions using final models
    print("\nMaking predictions with final models...")
    for c in competencies:
        print(f"Predicting C{c}...")
        mask_valid = df_test_final[f"C{c}"].notna()
        preds = pd.Series([pd.NA] * len(df_test_final), dtype="Int64")
        
        if mask_valid.any():
            y_pred = predict_scores(
                df_test_final.loc[mask_valid],
                tokenizers_final[c],
                models_final[c],
                c,
                text_col=text_col,
                max_len=max_len,
                batch_size=batch_size,
                device=device
            )
            preds.loc[mask_valid] = pd.Series(y_pred, index=df_test_final.index[mask_valid], dtype="Int64")
        
        out_final[f"pred_C{c}"] = preds
    
    # Save predictions if path provided
    if save_csv_path:
        out_final.to_csv(save_csv_path, index=False)
        print(f"✓ Final predictions saved to: {save_csv_path}")
    
    return out_final
