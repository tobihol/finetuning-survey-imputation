import wandb
from llm_survey_prediction.model_wrappers import (
    ClassificationWrapper,
    FinetuningClassifier,
    BERTClassifier,
)
from sklearn import metrics
import pandas as pd
from datetime import datetime
from typing import Iterable, Callable
import hashlib
import json
from catboost import CatBoostClassifier


def dict_hash(dictionary):
    dhash = hashlib.md5()
    encoded = json.dumps(
        dictionary,
        sort_keys=True,
        # make sets json serializable as lists
        default=lambda x: list(x) if isinstance(x, set) else x,
    ).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def classification_experiment(
    dataset_splits: Iterable,
    model_init_func: Callable,
    exp_config: dict,
    wandb_config: dict,
):
    """Run a classification experiment.

    Args:
        dataset_splits (Iterable): An iterable of dataset splits, where each split contains
            ((X_llm_train, y_llm_train, X_llm_test, y_llm_test),
             (X_trad_train, y_trad_train, X_trad_test, y_trad_test))
        model_init_func (Callable): Function that returns (models_trad, models_llm) tuple
            containing traditional and LLM model instances to evaluate
        exp_config (dict): Configuration dictionary for the experiment
        wandb_config (dict): Configuration dictionary for Weights & Biases logging
    """
    for split_idx, (llm_data, trad_data) in enumerate(dataset_splits):
        print(f"------ Split {split_idx} ------")
        run_config = exp_config | {
            "split_idx": split_idx,
        }
        models_trad, models_llm = model_init_func()

        # RUNS
        X_trad_train, y_trad_train, X_trad_test, y_trad_test = trad_data
        for model in models_trad:
            classification_run(
                model,
                X_train=X_trad_train,
                y_train=y_trad_train,
                X_test=X_trad_test,
                y_test=y_trad_test,
                wandb_config=wandb_config,
                run_config=run_config,
            )
        X_llm_train, y_llm_train, X_llm_test, y_llm_test = llm_data
        for model in models_llm:
            classification_run(
                model,
                X_train=X_llm_train,
                y_train=y_llm_train,
                X_test=X_llm_test,
                y_test=y_llm_test,
                wandb_config=wandb_config,
                run_config=run_config,
            )

        if run_config["debug"]:
            break


def classification_run(
    model: ClassificationWrapper,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    wandb_config: dict | None = None,
    run_config: dict | None = None,
) -> None:
    assert run_config if wandb_config else True, (
        "`run_config` must be provided if `wandb_config` is provided"
    )
    model_name = model.__class__.__name__
    print(f"Running {model_name}...")
    run_name = f"{model_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model_config = {"model_name": model_name} | model.get_params()
    model_config_hash = dict_hash(model_config)
    if wandb_config:
        wandb.init(
            **wandb_config,
            name=run_name,
            config={
                "run_config": run_config,
                "model_config": model_config,
                "model_config_hash": model_config_hash,
            },
            tags=[
                model_name,
                run_config["experiment"],
                f"model_config_hash_{model_config_hash}",
            ],
        )
    else:
        wandb.init(mode="disabled")
    if isinstance(model, CatBoostClassifier):
        model.fit(X_train, y_train, cat_features=list(X_train.columns), verbose=False)
    else:
        model.fit(X_train, y_train)
    if isinstance(model, FinetuningClassifier) or isinstance(model, BERTClassifier):
        y_proba = model.predict_proba(X_test, y_test)
    else:
        y_proba = model.predict_proba(X_test)
    y_proba_df = pd.DataFrame(y_proba, columns=model.classes_)
    y_pred = list(y_proba_df.idxmax(axis=1))

    wandb.log(
        data={
            "results": wandb.Table(
                data=pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
            ),
            "y_proba": wandb.Table(data=y_proba_df),
        }
    )

    wandb.summary["accuracy"] = metrics.accuracy_score(y_test, y_pred)
    wandb.summary["balanced_accuracy"] = metrics.balanced_accuracy_score(y_test, y_pred)
    wandb.summary["f1_macro"] = metrics.f1_score(y_test, y_pred, average="macro")
    wandb.summary["f1_micro"] = metrics.f1_score(y_test, y_pred, average="micro")
    wandb.summary["f1_weighted"] = metrics.f1_score(y_test, y_pred, average="weighted")

    wandb.finish()
