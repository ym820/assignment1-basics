
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataConfig:
    # Token ID paths
    train_data_path: str = "outputs/training/tinystories_train.uint16.bin"
    val_data_path: str = "outputs/training/tinystories_valid.uint16.bin"

    context_length: int = 256
    batch_size: int = 128

@dataclass
class ModelConfig:
    vocab_size: int = 10000
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: int = 10000

@dataclass
class OptimizerConfig:
    min_lr: Optional[float] = None
    max_lr: float = 1e-3

    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1

@dataclass
class ScheduleConfig:
    warmup_iters: int = 300
    cosine_cycle_iters: Optional[int] = None

@dataclass
class TrainingConfig:
    max_grad_norm: float = 1.0
    device: str = "auto"
    seed: int = 42

    total_tokens: int = 327680000
    max_steps: Optional[int] = None

    log_interval: int = 50
    eval_interval: int = 500
    eval_iters: int = 20
    checkpoint_interval: int = 1000
    resume_path: Optional[str] = None

@dataclass
class RunConfig:
    run_dir: str = "runs"
    prefix: str = "ts_baseline"
    name: Optional[str] = None

    enable_wandb: bool = True
    wandb_project: str = "cs336-a1"

@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    run: RunConfig = field(default_factory=RunConfig)

    
def resolve_config(cfg: TrainConfig) -> TrainConfig:
    assert cfg.model.d_model % cfg.model.num_heads == 0, f"d_model ({cfg.model.d_model}) must be divisible by num_heads ({cfg.model.num_heads})"

    if cfg.training.max_steps is None:
        # Calculate max_steps based on total token budget, batch size, and context length
        cfg.training.max_steps = cfg.training.total_tokens // (cfg.data.batch_size * cfg.data.context_length)
    assert cfg.schedule.warmup_iters < cfg.schedule.cosine_cycle_iters, f"warmup_iters ({cfg.schedule.warmup_iters}) must be less than cosine_cycle_iters ({cfg.schedule.cosine_cycle_iters})"

    if cfg.schedule.cosine_cycle_iters is None:
        # Set cosine cycle iters to the max steps
        cfg.schedule.cosine_cycle_iters = cfg.training.max_steps
    
    if cfg.optimizer.min_lr is None:
        # Set min_lr to 10% of max_lr
        cfg.optimizer.min_lr = 0.1 * cfg.optimizer.max_lr

    return cfg