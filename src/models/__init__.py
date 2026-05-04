"""Models module initialization."""

from .task1_lstm_ae import LSTMAutoencoder, LSTMEncoder, LSTMDecoder, reconstruction_loss
from .task2_vae import VariationalAutoencoder, VAEEncoder, VAEDecoder, vae_loss
from .task3_transformer import MusicTransformer, cross_entropy_loss, compute_perplexity
from .task4_rlhf import RewardModel, RLHFTrainer, HeuristicRewardFunction
from .baselines import RandomNoteGenerator, MarkovChainMusicModel

__all__ = [
    'LSTMAutoencoder',
    'VariationalAutoencoder',
    'MusicTransformer',
    'RewardModel',
    'RLHFTrainer',
    'RandomNoteGenerator',
    'MarkovChainMusicModel'
]
