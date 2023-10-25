from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint
from typing import Optional

import wandb
from model import (
    DEPRECATED_KEYS,
    Settings,
    WandbMetrics,
    train_segment,
    general_configuration,
)
from transformers import HfArgumentParser


@dataclass
class ArgumentSettings:
    sweep_id: Optional[str] = field(
        default="b6dmqiv3",  #  0.5065796375274658 large+add2+pseudo
        meta_information={"info": "ID for Wandb Sweep to obtain parameters"},
    )
    cublas_status: Optional[bool] = field(
        default=False,
        meta_information={
            "info": "Avoid locking CUBLAS (may impact experiment replication)"
        },
    )


def main():
    arg_parser = HfArgumentParser((ArgumentSettings, Settings))
    (command_line_args, training_settings) = arg_parser.parse_args_into_dataclasses()
    
    general_configuration(free_cublas=command_line_args.cublas_status)

    timer_start = datetime.now()
    sweep_identifier = command_line_args.sweep_id
    wandb_interface = wandb.Api()
    sweep_data = wandb_interface.run(f"{WandbMetrics.WANDB_DEBERTA_SWEEPS}/{sweep_identifier}")
    wandb_params = {param: value for param, value in sweep_data.config.items()}
    wandb_params["report_to"] = "wandb"
    
    for param, value in wandb_params.items():
        if param not in DEPRECATED_KEYS:
            if getattr(Settings, param) == getattr(training_settings, param):
                setattr(training_settings, param, value)

    pprint(training_settings)
    segment_outcomes, log_outcomes, output_directory = train_segment(config=training_settings)
    pprint(f"Execution duration: {datetime.now() - timer_start}")


if __name__ == "__main__":
    main()
