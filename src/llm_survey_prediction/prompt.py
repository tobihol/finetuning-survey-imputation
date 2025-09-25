import pandas as pd
import enum
import numpy as np


class InstructionPromptGLES2017(enum.StrEnum):
    basic = "The following is a description of a german voter, predict their vote.\n"
    extended = "Please perform a classification task. Given the survey answer from a national post election survey in germany, return which party the person voted for. Return a label from ['CDU', 'SPD', 'AFD', 'Grüne', 'Linke', 'FDP', 'andere Partei', 'Nichtwähler', 'keine Angabe'] only without any other text.\n"
    extended_no_missing = "Please perform a classification task. Given the survey answer from a national post election survey in germany, return which party the person voted for. Return a label from ['CDU', 'SPD', 'AFD', 'Grüne', 'Linke', 'FDP', 'andere Partei', 'Nichtwähler'] only without any other text.\n"
    system = "Please perform a classification task. Given the survey answers from a participant in a 2017 post-election survey in Germany, return the vote choice of the participant. Return a label from ['CDU', 'SPD', 'AFD', 'Grüne', 'Linke', 'FDP', 'andere Partei', 'Nichtwähler'] only without any other text."


class InstructionPromptGLES(enum.StrEnum):
    system = "Please perform a classification task. Given the survey answers from a participant in a post-election survey in Germany, return the vote choice of the participant. Return a label from ['CDU', 'SPD', 'AfD', 'Grüne', 'Linke', 'FDP', 'andere Partei', 'Nichtwähler'] only, without any other text."


class InstructionPromptANES2016(enum.StrEnum):
    system = "Please perform a classification task. Given the survey answers from a participant in a 2016 post-election survey in the US, return the vote choice of the participant. Return a label from ['Non-voter', 'Clinton', 'Trump'] only, without any other text."


def minimal_prompt(
    instruction: str,
    data: pd.DataFrame,
    label: pd.Series | None = None,
) -> pd.DataFrame:
    def create_prompt(row):
        prompt = instruction
        prompt += "\n".join([f"{k}: {v}" for k, v in row.items()])
        return prompt

    text_series = data.apply(create_prompt, axis="columns")

    return pd.DataFrame({"text": text_series, "label": label})


def prompt_w_system(
    system_prompt: str,
    X: pd.DataFrame,
    y: pd.Series | None = None,
) -> list[list[dict]]:
    prompts = []
    for idx in range(len(X)):
        row = X.iloc[idx]
        prompt = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": "\n".join([f"{k} {v}" for k, v in row.items()]),
            },
        ]
        if y is not None:
            label = y.iloc[idx]
            prompt.append(
                {
                    "role": "assistant",
                    "content": label,
                }
            )
        prompts.append(prompt)
    return prompts


def few_shot_prompt(
    system_prompt: str,
    X: pd.DataFrame,
    y: pd.Series | None = None,
    X_few_shot: pd.DataFrame | None = None,
    y_few_shot: pd.Series | None = None,
    n_few_shot: int = 5,
    seed: int = 24,
) -> list[list[dict]]:
    rng = np.random.RandomState(seed)
    prompts = []
    for idx in range(len(X)):
        row = X.iloc[idx]
        prompt = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        # few shot examples
        for _ in range(n_few_shot):
            shot_row_idx = rng.choice(len(X_few_shot))
            X_row = X_few_shot.iloc[shot_row_idx]
            y_row = y_few_shot.iloc[shot_row_idx]
            prompt.extend(
                [
                    {
                        "role": "user",
                        "content": "\n".join([f"{k} {v}" for k, v in X_row.items()]),
                    },
                    {
                        "role": "assistant",
                        "content": y_row,
                    },
                ]
            )

        # prompt for the target row
        prompt.append(
            {
                "role": "user",
                "content": "\n".join([f"{k} {v}" for k, v in row.items()]),
            },
        )
        if y is not None:
            label = y.iloc[idx]
            prompt.append(
                {
                    "role": "assistant",
                    "content": label,
                }
            )
        prompts.append(prompt)
    return prompts


LLAMA_INSTRUCT_CHAT_TEMPLATE = """{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""
