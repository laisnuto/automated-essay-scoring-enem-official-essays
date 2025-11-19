import os
from typing import Optional

def setup_colab_paths(
    base_path: Optional[str] = None,
    mount_drive: bool = True
) -> str:
    if mount_drive:
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
        except ImportError:
            print("⚠️ Google Colab not detected. Skipping drive mount.")
        except Exception as e:
            print(f"⚠️ Error mounting drive: {e}")
    
    if base_path is None:
        base_path = "/content/drive/MyDrive/enem_tcc_resultados"
    
    os.makedirs(base_path, exist_ok=True)
    print(f"✓ Base path: {base_path}")
    
    return base_path


def get_save_dir(base_path: str, subdir: str) -> str:
    save_dir = os.path.join(base_path, subdir)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


MODEL_TEMPLATES = {
    "jbsc_finetuned": {
        "mbert": "laisnuto/finetuning-with-official-enem-essays-jbsc-mbert-C{}",
        "bertugues": "laisnuto/finetuning-with-official-enem-essays-jbsc-bertugues-C{}",
        "bertimbau": "laisnuto/finetuning-with-official-enem-essays-jbsc-bertimbau-C{}"
    },
    "jbsc_original": {
        "bert-base": "kamel-usp/jbcs2025_bert-base-multilingual-cased-encoder_classification-C{}-essay_only",
        "bertugues": "kamel-usp/jbcs2025_BERTugues-base-portuguese-cased-encoder_classification-C{}-essay_only",
        "bertimbau": "kamel-usp/jbcs2025_bertimbau_base-C{}"
    },
    "jbsc_finetuned_by_comp": {
        "bertimbau": "kamel-usp/jbcs2025_bertimbau_base-C{}",
        "bertugues": "kamel-usp/jbcs2025_BERTugues-base-portuguese-cased-encoder_classification-C{}-essay_only",
        "mbert": "kamel-usp/jbcs2025_bert-base-multilingual-cased-encoder_classification-C{}-essay_only"
    },
    "originals_finetuned": {
        "mbert": "laisnuto/finetuning-with-official-enem-essays-mbert-C{}",
        "bertugues": "laisnuto/finetuning-with-official-enem-essays-bertugues-C{}",
        "bertimbau": "laisnuto/finetuning-with-official-enem-essays-bertimbau-C{}"
    }
}

TOKENIZER_NAMES = {
    "bertimbau": "neuralmind/bert-base-portuguese-cased"
}

COMPETENCIES = [1, 2, 3, 4, 5]
TEST_YEARS = [2016, 2018, 2022, 2023]
TRAIN_YEARS = [2019, 2020, 2021, 2024]

DEFAULT_HYPERPARAMS = {
    'learning_rate': 2e-5,
    'batch_size': 16,
    'epochs': 5
}
