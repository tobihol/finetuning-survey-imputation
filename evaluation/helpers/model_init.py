from llm_survey_prediction.model_wrappers import FinetuningClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

import itertools
from peft import LoraConfig, TaskType
from functools import partial
from llm_survey_prediction.prompt import prompt_w_system, InstructionPromptGLES2017


def init_models(SEED=24):
    # LLM models
    finetuning_models = []
    for (
        model_id,
        batch_size,
        n_epochs,
        quantization_config,
        lora_config,
    ) in itertools.product(
        # models
        [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        # batch sizes
        [
            1,
        ],
        # n_epochs
        [
            3,
        ],
        # quantization config
        [
            None
        ],
        # lora config
        [
            LoraConfig(
                r=256,  # change to the max that is in vram capacity
                lora_alpha=8,
                use_rslora=True,  # see https://arxiv.org/abs/2312.03732
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules="all-linear",
            ),
            # None,
        ],
    ):
        finetuning_model = FinetuningClassifier(
            model_id=model_id,
            prompt_func=partial(
                prompt_w_system, system_prompt=InstructionPromptGLES2017.system.value
            ),
            batch_size=batch_size,
            n_epochs=n_epochs,
            quantization_config=quantization_config,
            lora_config=lora_config,
            random_state=SEED,
        )
        finetuning_models.append(finetuning_model)

    models_llm = [
        CatBoostClassifier(random_state=SEED),
    ] + finetuning_models

    # Traditional models
    models_trad = [
        RandomForestClassifier(random_state=SEED),
        LogisticRegression(
            random_state=SEED,
            max_iter=1000,
        ),
    ]

    return (models_trad, models_llm)


def main():
    models_trad, models_llm = init_models(SEED=42)
    print("Number of traditional models:", len(models_trad))
    print("Number of LLM-based models:", len(models_llm))
    print("Traditional models:", [model.get_params() for model in models_trad])
    print("LLM-based models:", [model.get_params() for model in models_llm])

if __name__ == "__main__":
    main()
