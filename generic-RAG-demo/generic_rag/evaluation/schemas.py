from pydantic import BaseModel, Field
from pathlib import Path
import yaml


def load_or_extract_config(config_dict: dict = None, config_path: Path = None, default_path: Path = None) -> dict:
    """
    Helper to load config from dict or file path, reducing duplication.

    Args:
        config_dict: Optional preloaded config dictionary
        config_path: Optional path to config file
        default_path: Default config path if neither config_dict nor config_path provided

    Returns:
        Dict containing the loaded configuration

    Raises:
        ValueError: If no valid config source is provided
    """
    if config_dict is not None:
        return config_dict

    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if default_path is not None:
        with open(default_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    raise ValueError("Must provide either config_dict, config_path, or default_path")


def create_judge_schema(config: dict) -> type:
    """Create a dynamic Pydantic schema based on required preloaded config."""

    eval_cfg = config["evaluation"]
    llm_judge_metrics = eval_cfg["metrics"]["llm_judge"]
    scale_min = eval_cfg["scoring"]["scale_min"]
    scale_max = eval_cfg["scoring"]["scale_max"]

    # Build dynamic fields dictionary for class creation
    fields = {}
    annotations = {}

    for metric in llm_judge_metrics:
        annotations[metric] = int
        fields[metric] = Field(description=f"Score {scale_min}-{scale_max}")

    # Create dynamic class with proper annotations and fields
    class_dict = {"__annotations__": annotations, **fields}

    return type("DynamicJudgeSchema", (BaseModel,), class_dict)


# Default schema for backward compatibility
class JudgeSchema(BaseModel):
    correctness: int = Field(description="Score 0-5")
    relevance: int = Field(description="Score 0-5")
    conciseness: int = Field(description="Score 0-5")
    faithfulness: int = Field(description="Score 0-5")
