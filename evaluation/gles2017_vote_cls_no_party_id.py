from helpers.eval_pipeline import classification_experiment
from helpers.data_preprocessing import DatasetGLES2017NoPartyId
from helpers.model_init import init_models
from sklearn.model_selection import StratifiedKFold, BaseCrossValidator
from pathlib import Path
from datetime import datetime
from functools import partial

# CONFIG

SEED = 24

WANDB_CONFIG = {
    # INSERT WANDB ACCOUNT
    # "entity": "username",
    # "project": "project-name",
}

DATASET = DatasetGLES2017NoPartyId(
    path="datasets/ZA6835_v1-0-0.sav",
)
TARGET_COL = "Wahlentscheidung"

CROSS_VALIDATOR = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

DEBUG = False
# CONFIG END

assert isinstance(CROSS_VALIDATOR, BaseCrossValidator)

# experiment config for logging. (Models are logged separately.)
exp_config = {
    "experiment": f"{Path(__file__).stem}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",  # name of this file
    "cross_validation_config": {
        "validator_name": CROSS_VALIDATOR.__class__.__name__,
        **CROSS_VALIDATOR.__getstate__(),
    },
    "seed": SEED,
    "debug": DEBUG,
}
# if DEBUG:
#     DATASET._df_raw = DATASET._df_raw[:20]  # small sample for testing

splits_gen = DATASET.classification_splits(
    target_col=TARGET_COL,
    splits=CROSS_VALIDATOR,
)

# EVALUATION
classification_experiment(
    dataset_splits=splits_gen,
    model_init_func=partial(init_models, SEED=SEED),
    exp_config=exp_config,
    wandb_config=WANDB_CONFIG,
)