#!/usr/bin/python

"""
Temporary configs for conformer generation.
Code adapted from https://github.com/deepmind/flows_for_atomic_solids 
"""

from logging import Logger
import torch
from ml_collections import config_dict
from torch.nn import Transformer
from models.base import Base
from models.conditioner import Conditioner
from models.model import make_model
from models.flows import make_split_coupling_flow
from models.transforms import Dihedral2Coord
from models.energy import Energy


FREQUENCIES = {
    4: 8,
    8: 8,
    14: 8,
    16: 8,
    32: 8,
}


def get_config(num_atoms: int, logger: Logger):
    """Returns the config."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frequencies = FREQUENCIES[num_atoms]
    train_batch_size = 128
    config = config_dict.ConfigDict()
    config.state = dict(
        num_atoms=num_atoms,
        beta=2,
        lower=-torch.pi,
        upper=torch.pi,
    )
    conditioner = dict(
        constructor=Conditioner,
        kwargs=dict(
            embedding_size=256,
            num_frequencies=num_frequencies,
            conditioner_constructor=Transformer,
            conditioner_kwargs=dict(
                nhead=2,
                num_encoder_layers=2,
                num_decoder_layers=2,
                dropout=0.,))
    )
    config.model = dict(
        constructor=make_model,
        kwargs=dict(
            bijector=dict(
                constructor=make_split_coupling_flow,
                kwargs=dict(
                    num_layers=24,
                    num_bins=16,
                    conditioner=conditioner,
                    use_circular_shift=False,
                ),
            ),
            base=dict(
                constructor=Base,
                kwargs=dict(
                    num_atoms=num_atoms,
                    batch_size=train_batch_size,
                ),
            ),
            coord_trans=dict(
                constructor=Dihedral2Coord,
            ),
            logger=logger,
            device=device
        ),
    )
    config.energy = Energy.apply
    config.train = dict(
        batch_size=train_batch_size,
        learning_rate=7e-5,
        learning_rate_decay_steps=[250000, 500000],
        learning_rate_decay_factor=0.1,
        patience=4,
        seed=42,
        max_gradient_norm=10000.,
    )
    config.test = dict(
        test_every=50,
        save_threshold=310,
        batch_size=train_batch_size,
    )
    return config
