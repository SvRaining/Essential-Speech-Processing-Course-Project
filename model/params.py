from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint
from typing import Optional

import wandb
from model import (
    SAVE_TOP_MODELS,
    Settings,
    WandbMetrics,
    train_segment,
    general_configuration,
    fetch_tags,
)
from transformers import HfArgumentParser

BASIC_PARAM_SET = {
    "strategy": "arbitrary",
    "annotation": "|".join(fetch_tags()),
    "evaluation": {
        "label": "nano.test_error_rate",
        "objective": "reduce",
    },
    "factors": {
        "init_value": {
            "pattern": "integer_range",
            "least": 0,
            "most": 1000,
        },
        "speed_factor": {
            "pattern": "log_set",
            "least": 5e-7,
            "most": 1e-4,
        },
        "hide_factor": {
            "pattern": "log_set",
            "least": 5e-4,
            "most": 5e-2,
        },
        "focus_factor": {
            "pattern": "log_set",
            "least": 5e-4,
            "most": 5e-2,
        },
        "weight_reduction": {
            "pattern": "log_set",
            "least": 5e-5,
            "most": 1e-2,
        },
        "set_size": {"amount": 8},
    },
}

ADVANCED_PARAM_SET = {
    "strategy": "arbitrary",
    "annotation": "|".join(fetch_tags()),
    "evaluation": {
        "label": "nano.test_error_rate",
        "objective": "reduce",
    },
    "factors": {
        "init_value": {
            "pattern": "integer_range",
            "least": 0,
            "most": 1000,
        },
        "speed_factor": {
            "pattern": "log_set",
            "least": 1e-6,
            "most": 5e-5,
        },
        "hide_factor": {
            "pattern": "log_set",
            "least": 5e-5,
            "most": 5e-2,
        },
        "focus_factor": {
            "pattern": "log_set",
            "least": 5e-5,
            "most": 5e-2,
        },
        "weight_reduction": {
            "pattern": "log_set",
            "least": 5e-5,
            "most": 1e-2,
        },
        "set_size": {"amount": 6},
    },
}

EXTREME_PARAM_SET = {
    "strategy": "arbitrary",
    "annotation": "|".join(fetch_tags()),
    "evaluation": {
        "label": "nano.test_error_rate",
        "objective": "reduce",
    },
    "factors": {
        "init_value": {
            "pattern": "integer_range",
            "least": 0,
            "most": 1000,
        },
        "speed_factor": {
            "pattern": "log_set",
            "least": 1e-6,
            "most": 1e-4,
        },
        "hide_factor": {
            "pattern": "log_set",
            "least": 5e-5,
            "most": 5e-2,
        },
        "focus_factor": {
            "pattern": "log_set",
            "least": 5e-5,
            "most": 5e-2,
        },
        "weight_reduction": {
            "pattern": "log_set",
            "least": 5e-5,
            "most": 1e-2,
        },
        "static_layers": {"levels": [3, 6, 9]},
    },
}

@dataclass
class ArgumentPrompt:
    scan_identifier: Optional[str] = field(
        default=None,
        metadata={"help": "ID for Wandb Scan. If omitted, new scan initialized."},
    )
    cublas_toggle: Optional[bool] = field(
        default=False,
        metadata={"help": "Disable CUBLAS toggle (can alter experiment outcomes)"},
    )
    iteration_count: Optional[int] = field(
        default=SAVE_TOP_MODELS,
        metadata={"help": "Count of hyperparameter search iterations"},
    )
    retain_top_experiments: Optional[int] = field(
        default=SAVE_TOP_MODELS,
        metadata={"help": "Count of experiments to retain"},
    )

if __name__ == "__main__":
    arg_fetcher = HfArgumentParser((ArgumentPrompt, Settings))
    (user_input, training_vars) = arg_fetcher.parse_args_into_dataclasses()
    general_configuration(free_cublas=user_input.cublas_toggle)

    inception = datetime.now()
    if user_input.scan_identifier is None:
        if "xtreme" in training_vars.model_identifier:
            scan_vars = EXTREME_PARAM_SET.copy()
        elif "adv" in training_vars.model_identifier:
            scan_vars = ADVANCED_PARAM_SET.copy()
        elif "basic" in training_vars.model_identifier:
            scan_vars = BASIC_PARAM_SET.copy()
        else:
            raise ValueError(f"Unsupported model label: {training_vars.model_identifier}")
        scan_identifier = wandb.sweep(scan_vars, project=WandbMetrics.WANDB_DEBERTA_SWEEPS)
    else:
        scan_identifier = user_input.scan_identifier

    wandb.worker(
        scan_id=scan_identifier,
        function=lambda: train_segment(
            input_vars=training_vars, retain_top_experiments=user_input.retain_top_experiments
        ),
        iteration_count=user_input.iteration_count,
        project=WandbMetrics.WANDB_DEBERTA_SWEEPS,
    )
    pprint(f"Execution clock: {datetime.now() - inception}")
