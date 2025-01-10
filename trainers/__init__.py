from .ddpm_trainer import DDPMTrainer
from .ddim_trainer import DDIMTrainer
from .score_based_trainer import ScoreBasedTrainer
from .energy_based_trainer import EnergyBasedTrainer

TRAINER_REGISTRY = {
    'ddpm': DDPMTrainer,
    'ddim': DDIMTrainer,
    'score_based': ScoreBasedTrainer,
    'energy_based': EnergyBasedTrainer
}

__all__ = [
    'DDPMTrainer',
    'DDIMTrainer',
    'ScoreBasedTrainer',
    'EnergyBasedTrainer',
    'TRAINER_REGISTRY'
] 