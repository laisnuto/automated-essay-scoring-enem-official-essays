import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    root_mean_squared_error,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import ast

PROTOCOL_LABELS = {
    "no_change": "Sem ajuste de escala",
    "dup_bounds": "Correção dupla (baixo/cima)",
    "truth_floor40": "Arred. verdade p/ baixo (40)",
    "truth_ceil40": "Arred. verdade p/ cima (40)",
    "only_true_mult40": "Apenas verdade múltipla de 40",
}

PROTOCOL_ORDER = [
    "no_change",
    "dup_bounds",
    "truth_floor40",
    "truth_ceil40",
    "only_true_mult40"
]


def arredondar_notas(notas: List[int]) -> List[int]:
    referencia = [0, 40, 80, 120, 160, 200]
    novas_notas = []
    for n in notas:
        mais_prox = 1000
        arredondado = -1
        for r in referencia:
            if abs(n - r) < mais_prox:
                arredondado = r
                mais_prox = abs(n - r)
        novas_notas.append(arredondado)
    return novas_notas


def calcular_div(notas1: List[int], notas2: List[int]) -> float:
    """Calculate horizontal divergence: two scores are divergent if difference > 80."""
    div = 0
    for n1, n2 in zip(notas1, notas2):
        if abs(n1 - n2) > 80:
            div += 1
    return 100 * div / len(notas1) if len(notas1) > 0 else 0.0


def calcular_agregado(dic_perf: Dict[str, float]) -> float:
    acc = dic_perf['ACC'] * 100
    rmse = (200 - dic_perf['RMSE']) / 2
    qwk = dic_perf['QWK'] * 100
    div = 100 - dic_perf['DIV']
    return (acc + rmse + qwk + div) / 4


def calcular_resultados(
    y: List[int],
    y_hat: List[int],
    is_final: bool = False,
    qwk_step: int = 20
) -> Dict:
    if is_final:
        ALL_LABELS = list(range(0, 1001, qwk_step))
    else:
        ALL_LABELS = list(range(0, 201, qwk_step))
    
    ACC = accuracy_score(y, y_hat)
    RMSE = root_mean_squared_error(y, y_hat)
    QWK = cohen_kappa_score(y, y_hat, weights='quadratic', labels=ALL_LABELS)
    DIV = calcular_div(y, y_hat)
    macro_f1 = f1_score(y, y_hat, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, y_hat, average="weighted", zero_division=0)
    
    dic = {
        'ACC': ACC,
        'RMSE': RMSE,
        'QWK': QWK,
        'DIV': DIV,
        'F1-Macro': macro_f1,
        'F1-Weighted': weighted_f1,
        'y': y,
        'y_hat': y_hat,
        'Agregado': calcular_agregado({
            'ACC': ACC,
            'RMSE': RMSE,
            'QWK': QWK,
            'DIV': DIV
        }),
    }
    
    return dic


def ajustar_para_correcao_dupla(
    y_true: List[int],
    y_pred: List[int]
) -> Tuple[List[int], List[int]]:
    """
    Adjust for double correction protocol:
    - If truth is multiple of 40 -> duplicate (r,r) and (p,p)
    - Otherwise -> create (low, high) for truth and duplicate p
    """
    y_true_adj, y_pred_adj = [], []
    for r, p in zip(y_true, y_pred):
        if pd.isna(r) or pd.isna(p):
            continue
        r = int(r)
        p = int(p)
        if r % 40 == 0:
            y_true_adj.extend([r, r])
            y_pred_adj.extend([p, p])
        else:
            baixo = (r // 40) * 40
            cima = baixo + 40
            y_true_adj.extend([baixo, cima])
            y_pred_adj.extend([p, p])
    return y_true_adj, y_pred_adj


def arredonda_verdade(y_true: List[int], modo: str) -> List[int]:
    y_true = pd.Series(y_true).dropna().astype(int)
    if modo == 'floor':
        return (np.floor(y_true / 40) * 40).astype(int).tolist()
    elif modo == 'ceil':
        return (np.ceil(y_true / 40) * 40).astype(int).tolist()
    elif modo == 'none':
        return y_true.tolist()
    else:
        raise ValueError("modo inválido")


def filtra_verdades_multiplas_40(
    y_true: List[int],
    y_pred: List[int]
) -> Tuple[List[int], List[int]]:
    mask = (pd.Series(y_true).astype(int) % 40 == 0)
    y_true_f = pd.Series(y_true)[mask].astype(int).tolist()
    y_pred_f = pd.Series(y_pred)[mask].astype(int).tolist()
    return y_true_f, y_pred_f


def apply_protocol(
    y_true: List[int],
    y_pred: List[int],
    protocol: str
) -> Tuple[List[int], List[int]]:
    """Apply evaluation protocol to align true and predicted scores."""
    if protocol == "no_change":
        return y_true, y_pred
    elif protocol == "dup_bounds":
        return ajustar_para_correcao_dupla(y_true, y_pred)
    elif protocol == "truth_floor40":
        y_r = arredonda_verdade(y_true, "floor")
        return y_r, y_pred
    elif protocol == "truth_ceil40":
        y_r = arredonda_verdade(y_true, "ceil")
        return y_r, y_pred
    elif protocol == "only_true_mult40":
        return filtra_verdades_multiplas_40(y_true, y_pred)
    else:
        raise ValueError(f"Protocol desconhecido: {protocol}")


def load_enem_dataset(
    dataset_name: str = "laisnuto/self-collected-ENEM-dataset",
    split: str = "train",
    anos_teste: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset
    
    if anos_teste is None:
        anos_teste = [2016, 2018, 2022, 2023]
    
    print("Carregando o dataset...")
    dataset = load_dataset(dataset_name, split=split)
    df = dataset.to_pandas()
    
    def _to_list(x):
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            return [None] * 5
    
    notas_expandidas = df["notas"].apply(_to_list)
    df[["C1", "C2", "C3", "C4", "C5"]] = pd.DataFrame(
        notas_expandidas.tolist(), index=df.index
    )
    
    for c in ["C1", "C2", "C3", "C4", "C5"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    
    anos_treino = [2019, 2020, 2021, 2024]
    
    df_train = df[df["ano"].isin(anos_treino)].reset_index(drop=True)
    df_test = df[df["ano"].isin(anos_teste)].reset_index(drop=True)
    
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
    
    return df_train, df_test


def plot_real_vs_predicted(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    comp_key: str,
    model_name: str,
    save_path: Optional[str] = None,
    qwk_step: int = 20
) -> Tuple[float, float]:
    resultado = calcular_resultados(
        y_real.tolist(), y_pred.tolist(), qwk_step=qwk_step
    )
    qwk = resultado['QWK']
    
    X = y_pred.reshape(-1, 1)
    y = y_real
    reg = LinearRegression()
    reg.fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = reg.score(X, y)
    
    x_line = np.array([0, 200])
    y_line = slope * x_line + intercept
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    np.random.seed(42)
    jitter_x = np.random.normal(0, 1.5, size=len(y_pred))
    jitter_y = np.random.normal(0, 1.5, size=len(y_real))
    
    ax.scatter(
        y_pred + jitter_x, y_real + jitter_y,
        alpha=0.5, s=30, edgecolors='black', linewidths=0.3
    )
    
    ax.plot([0, 200], [0, 200], 'r--', linewidth=2, label='y = x', alpha=0.8)
    ax.plot(
        x_line, y_line, 'b-', linewidth=2,
        label=f'Regressão: y = {slope:.3f}x + {intercept:.2f}', alpha=0.8
    )
    
    ax.set_xlabel('Predicted Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Real Score', fontsize=12, fontweight='bold')
    ax.set_title(
        f'{comp_key}\nR² = {r2:.4f} | QWK = {qwk:.4f}',
        fontsize=13, fontweight='bold'
    )
    
    ax.set_xlim(-10, 210)
    ax.set_xticks(range(0, 201, 40))
    ax.set_xticklabels(range(0, 201, 40), rotation=45, ha='right', fontsize=9)
    
    ax.set_ylim(-10, 210)
    ax.set_yticks(range(0, 201, 20))
    ax.set_yticklabels(range(0, 201, 20), fontsize=9)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return r2, qwk


def plot_confusion_matrix(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    comp_key: str,
    model_name: str,
    save_path: Optional[str] = None,
    qwk_step: int = 20
) -> Dict[str, float]:
    def arredondar_para_multiplo_20(nota: int) -> int:
        return int(round(nota / 20) * 20)
    
    y_real_clean = np.array([arredondar_para_multiplo_20(n) for n in y_real])
    y_pred_clean = np.array([arredondar_para_multiplo_20(n) for n in y_pred])
    
    y_real_clean = np.clip(y_real_clean, 0, 200)
    y_pred_clean = np.clip(y_pred_clean, 0, 200)
    
    labels_possiveis = list(range(0, 201, 20))
    cm = confusion_matrix(y_real_clean, y_pred_clean, labels=labels_possiveis)
    
    resultado = calcular_resultados(
        y_real_clean.tolist(), y_pred_clean.tolist(), qwk_step=qwk_step
    )
    qwk = resultado['QWK']
    accuracy = accuracy_score(y_real_clean, y_pred_clean)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_real_clean, y_pred_clean, average='weighted', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_real_clean, y_pred_clean, average='macro', zero_division=0
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(
        f'Confusion Matrix - {comp_key} - Model {model_name}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels_possiveis, yticklabels=labels_possiveis,
        ax=ax, cbar_kws={'label': 'Count'},
        linewidths=0.5, linecolor='gray'
    )
    
    ax.set_xlabel('Predicted Grade', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Grade', fontsize=12, fontweight='bold')
    ax.set_title(f'{comp_key}', fontsize=14, fontweight='bold')
    
    metricas_texto = (
        f'Acc: {accuracy:.3f} | QWK: {qwk:.3f} | '
        f'F1-W: {f1:.3f} | F1-M: {f1_macro:.3f}'
    )
    ax.text(
        0.5, -0.12, metricas_texto, transform=ax.transAxes,
        ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return {
        'accuracy': accuracy,
        'qwk': qwk,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'total_samples': len(y_real_clean)
    }


def generate_latex_table(
    avaliacoes_por_modelo: Dict,
    model_key: str,
    metric_key: str,
    metric_title: str,
    competencias: List[int] = [1, 2, 3, 4, 5]
) -> str:
    df_tab = pd.DataFrame(
        index=[PROTOCOL_LABELS[k] for k in PROTOCOL_ORDER],
        columns=[f"C{i}" for i in competencias],
        dtype=float
    )
    
    protocolos = avaliacoes_por_modelo.get(model_key, {})
    for sk in PROTOCOL_ORDER:
        if sk not in protocolos:
            continue
        compdict = protocolos[sk]
        for c in competencias:
            ck = f"C{c}"
            if ck in compdict and metric_key in compdict[ck]:
                df_tab.loc[PROTOCOL_LABELS[sk], ck] = compdict[ck][metric_key]
    
    df_print = df_tab.round(3)
    caption = (
        f"{metric_title} por competência para o modelo {model_key} "
        f"nos diferentes protocolos de avaliação"
    )
    label = f"tab:{metric_key.replace('-','').replace(' ','').lower()}_{model_key.replace('-','_')}"
    
    return df_print.to_latex(
        index=True, caption=caption, label=label,
        na_rep="--", float_format="%.3f", escape=True
    )
