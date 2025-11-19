# ENEM TCC Resultados

Repositório com código otimizado para experimentos de avaliação automática de redações do ENEM usando modelos de linguagem.

## Estrutura do Projeto

```
enem_tcc_resultados/
├── model_utils.py          # Funções comuns relacionadas aos modelos
├── utils.py                # Funções de métricas, tabelas e gráficos
├── config.py               # Configurações e constantes
├── README.md               # Este arquivo
├── fine_tuning_modelos_jbsc/      # Notebooks de fine-tuning
├── fine_tuning_modelos_originais/ # Notebooks de fine-tuning originais
├── inferencia_fine_tuning_jbsc/   # Notebooks de inferência (fine-tuned)
├── inferencia_fine_tuning_originals/ # Notebooks de inferência (originais)
├── inferencia_modelos_jbsc/       # Notebooks de inferência (zero-shot)
└── grade_expectation/             # Notebooks de cálculo de nota esperada
```

## Módulos Utilitários

### `model_utils.py`
Contém funções relacionadas ao carregamento, treinamento e uso de modelos:
- `get_device()`: Detecta e retorna o dispositivo (GPU/CPU)
- `load_tokenizer_and_model()`: Carrega tokenizer e modelo
- `create_pipeline()`: Cria pipeline para inferência
- `train_model_cv()`: Treina modelo com validação cruzada
- `train_final_model()`: Treina modelo final
- `predict_scores()`: Faz predições com modelo treinado
- `calculate_expected_grade()`: Calcula nota esperada usando softmax
- `EnemCompDataset`: Classe Dataset para treinamento

### `utils.py`
Contém funções de avaliação, visualização e processamento:
- **Métricas**: `calcular_resultados()`, `calcular_div()`, `calcular_agregado()`
- **Protocolos**: `apply_protocol()`, `ajustar_para_correcao_dupla()`, etc.
- **Visualização**: `plot_real_vs_predicted()`, `plot_confusion_matrix()`
- **Tabelas**: `generate_latex_table()`
- **Dados**: `load_enem_dataset()`

### `config.py`
Contém configurações e constantes:
- `MODEL_TEMPLATES`: Templates de nomes de modelos
- `TOKENIZER_NAMES`: Nomes de tokenizers especiais
- `COMPETENCIES`: Lista de competências [1, 2, 3, 4, 5]
- `TEST_YEARS`, `TRAIN_YEARS`: Anos de teste e treino
- `setup_colab_paths()`: Configuração opcional para Google Colab

## Como Usar

### 1. Instalação de Dependências

```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn tqdm
```

### 2. Uso em Notebooks

Os notebooks foram refatorados para usar os módulos utilitários. Exemplo:

```python
# Imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(''))))

from model_utils import get_device, create_pipeline, TOKENIZER_NAMES
from utils import calcular_resultados, load_enem_dataset, PROTOCOL_ORDER
from config import setup_colab_paths, get_save_dir, MODEL_TEMPLATES, TEST_YEARS

# Setup (opcional para Google Colab)
try:
    BASE_PATH = setup_colab_paths(mount_drive=True)
except:
    BASE_PATH = os.path.join(os.getcwd(), "results")
    os.makedirs(BASE_PATH, exist_ok=True)

# Carregar dataset
_, df_test = load_enem_dataset(anos_teste=TEST_YEARS)

# Carregar modelos
device = get_device()
model_types = MODEL_TEMPLATES["jbsc_original"]
pipelines = {}
for model_key, template in model_types.items():
    # ... criar pipelines
```

### 3. Google Colab (Opcional)

O código funciona tanto localmente quanto no Google Colab. Se estiver no Colab, o `setup_colab_paths()` montará automaticamente o Google Drive. Caso contrário, usará caminhos locais.

## Protocolos de Avaliação

O módulo `utils.py` suporta 5 protocolos de avaliação:

1. **no_change**: Sem ajuste de escala
2. **dup_bounds**: Correção dupla (baixo/cima)
3. **truth_floor40**: Arredonda verdade para baixo (múltiplos de 40)
4. **truth_ceil40**: Arredonda verdade para cima (múltiplos de 40)
5. **only_true_mult40**: Apenas casos com verdade múltipla de 40

## Métricas Calculadas

- **ACC**: Acurácia
- **RMSE**: Root Mean Squared Error
- **QWK**: Quadratic Weighted Kappa
- **DIV**: Divergência horizontal
- **F1-Macro**: F1 score (macro)
- **F1-Weighted**: F1 score (weighted)
- **Agregado**: Métrica agregada

## Gerando Gráficos e Visualizações

Os notebooks geram automaticamente gráficos e matrizes de confusão quando executados. Esses arquivos são salvos localmente mas **não são versionados no Git** (veja `.gitignore`).

Para gerar as visualizações:
1. Execute os notebooks completos
2. Os gráficos serão salvos nas pastas correspondentes (ex: `inferencia_modelos_jbsc/`)
3. As imagens podem ser recriadas a qualquer momento executando os notebooks novamente

**Nota**: Imagens e CSVs grandes são ignorados pelo Git para manter o repositório leve. Se precisar compartilhar resultados específicos, considere usar um serviço como Google Drive ou criar um repositório separado para resultados.

## Contribuindo

Este repositório foi otimizado para ser público. Todos os notebooks foram refatorados para usar os módulos utilitários, removendo código duplicado e facilitando manutenção.

## Licença

[Adicione sua licença aqui]

