from pydantic import BaseModel, Field
from typing import Optional, List

class CurriculumBaseModel(BaseModel):
    start: int
    end: int
    inc: int
    interval: int

class CurriculumModel(BaseModel):
    dims: CurriculumBaseModel
    points: CurriculumBaseModel

class ModelConfig(BaseModel):
    family: str = Field(..., allowed=["gpt2", "lstm"])
    n_positions: int
    n_dims: int
    n_embd: int
    n_layer: int
    n_head: int

TASK_LIST = [
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
]

class TrainingConfig(BaseModel):
    task: str = Field(..., allowed=TASK_LIST)
    task_kwargs: dict
    num_tasks: Optional[int] = None
    num_training_examples: Optional[int] = None
    data: str = Field(..., allowed=["gaussian"])
    batch_size: int = 64
    learning_rate: float = 3e-4
    train_steps: int = 1000
    save_every_steps: int = 1000
    keep_every_steps: int = -1
    resume_id: Optional[str] = None
    curriculum: CurriculumModel

class WandbConfig(BaseModel):
    project: str = "in-context-training"
    entity: str = "in-context"
    notes: str = ""
    name: Optional[str] = None
    log_every_steps: int = 10

class MainConfig(BaseModel):
    out_dir: str
    model: ModelConfig
    training: TrainingConfig
    wandb: WandbConfig
    test_run: bool = False
