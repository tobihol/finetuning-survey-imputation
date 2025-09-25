# Learning from Convenience Samples: A Case Study on Fine-Tuning LLMs for Survey Non-response in the German Longitudinal Election Study

<!-- Code for the paper TODO. -->

To setup the environment, install the uv package manager and run: 

```bash
uv sync
```

Download the dataset `ZA6835_v1-0-0.sav` (https://doi.org/10.4232/1.13648) and it in the `datasets` folder.

Run the experiments:

```bash
python evaluation/gles2017_vote_cls.py
python evaluation/gles2017_vote_cls_no_party_id.py
python evaluation/gles2017_vote_cls_uni_and_school.py
python evaluation/gles2017_vote_cls_uni_and_school_no_id.py
python evaluation/gles2017_vote_cls_thueringen.py
python evaluation/gles2017_vote_cls_thueringen_no_id.py
python evaluation/gles2017_vote_cls_unemp.py
python evaluation/gles2017_vote_cls_unemp_no_id.py
```
(add a wandb user and project id to the evaluation scripts for logging)