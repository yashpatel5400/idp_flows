#!/usr/bin/python
#
# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Energy-based training of a flow model on an atomistic system."""

import sys
sys.path.append("../")

from typing import Callable, Dict, Tuple, Union

from absl import app
from absl import flags
import chex
import copy
import distrax
from idp_flows.experiments import lennard_jones_config
from idp_flows.experiments import monatomic_water_config
from idp_flows.experiments import utils
from idp_flows.utils import observable_utils as obs_utils
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from mpi4py import MPI
import mpi4jax

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Array = chex.Array
Numeric = Union[Array, float]

flags.DEFINE_enum('system', 'mw_cubic_8',
                  ['mw_cubic_8', 'mw_cubic_64', 'mw_cubic_216', 'mw_cubic_512',
                   'mw_hex_64', 'mw_hex_216', 'mw_hex_512',
                   'lj_32', 'lj_256', 'lj_500',
                   ], 'System and number of particles to train.')
flags.DEFINE_integer('num_iterations', int(10**6), 'Number of training steps.')


FLAGS = flags.FLAGS


def _num_particles(system: str) -> int:
  return int(system.split('_')[-1])


def _get_loss(
    model: distrax.Distribution,
    energy_fn: Callable[[Array], Array],
    beta: Numeric,
    num_samples: int) -> Tuple[Array, Dict[str, Array]]:
  """Returns the loss and stats."""
  rng_key = hk.next_rng_key()
  samples, log_prob = model.sample_and_log_prob(
      seed=rng_key, sample_shape=num_samples)
  energies = energy_fn(samples)
  energy_loss = jnp.mean(beta * energies + log_prob)

  loss = energy_loss
  stats = {
      'energy': energies,
      'model_log_prob': log_prob,
      'target_log_prob': -beta * energies
  }
  return loss, stats


def main(_):
  system = FLAGS.system
  if system.startswith('lj'):
    config = lennard_jones_config.get_config(_num_particles(system))
  elif system.startswith('mw_cubic'):
    config = monatomic_water_config.get_config(_num_particles(system), 'cubic')
  elif system.startswith('mw_hex'):
    config = monatomic_water_config.get_config(_num_particles(system), 'hex')
  else:
    raise KeyError(system)

  state = config.state
  energy_fn_train = config.train_energy.constructor(
      **config.train_energy.kwargs)
  energy_fn_test = config.test_energy.constructor(**config.test_energy.kwargs)

  lr_schedule_fn = utils.get_lr_schedule(
      config.train.learning_rate, config.train.learning_rate_decay_steps,
      config.train.learning_rate_decay_factor)
  optimizer = optax.chain(
      optax.scale_by_adam(),
      optax.scale_by_schedule(lr_schedule_fn),
      optax.scale(-1))
  if config.train.max_gradient_norm is not None:
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.train.max_gradient_norm), optimizer)

  def create_model():
    return config.model['constructor'](
        num_particles=state.num_particles,
        lower=state.lower,
        upper=state.upper,
        **config.model['kwargs'])

  def loss_fn():
    """Loss function for training."""
    model = create_model()

    loss, stats = _get_loss(
        model=model,
        energy_fn=energy_fn_train,
        beta=state.beta,
        num_samples=config.train.batch_size // size,
        )

    metrics = {
        'loss': loss,
        'energy': jnp.mean(stats['energy']),
        'model_entropy': -jnp.mean(stats['model_log_prob']),
    }
    return loss, metrics

  def eval_fn():
    """Evaluation function."""
    model = create_model()
    loss, stats = _get_loss(
        model=model,
        energy_fn=energy_fn_test,
        beta=state.beta,
        num_samples=config.test.batch_size,
        )
    log_probs = {
        'model_log_probs': stats['model_log_prob'],
        'target_log_probs': stats['target_log_prob'],
    }
    metrics = {
        'loss': loss,
        'energy': jnp.mean(stats['energy']),
        'model_entropy': -jnp.mean(stats['model_log_prob']),
        'ess': obs_utils.compute_ess(**log_probs),
        'logz': obs_utils.compute_logz(**log_probs),
        'logz_per_particle':
            obs_utils.compute_logz(**log_probs) / state.num_particles,
    }
    return metrics

  print(f'Initialising system {system}')
  rng_key = jax.random.PRNGKey(config.train.seed + rank) # each MPI node is seeded differently to benefit from parallelism
  init_fn, apply_fn = hk.transform(loss_fn)
  _, apply_eval_fn = hk.transform(eval_fn)

  rng_key, init_key = jax.random.split(rng_key)
  params = init_fn(init_key)
  opt_state = optimizer.init(params)

  def _loss(params, rng):
    loss, metrics = apply_fn(params, rng)
    return loss, metrics
  
  jitted_loss = jax.value_and_grad(_loss, has_aux=True) # jax.jit(jax.value_and_grad(_loss, has_aux=True))
  jitted_eval = apply_eval_fn # jax.jit(apply_eval_fn)

  step = 0
  print('Beginning of training.')
  while step < FLAGS.num_iterations:
    # Training update.
    rng_key, loss_key = jax.random.split(rng_key)
    (_, metrics), g = jitted_loss(params, loss_key)
    
    if rank == 0:
      if (step % 25) == 0:
        print(f'Train[{step}]: {metrics}')

      for i in range(1, size):
        data = comm.recv(source=i)

        for layer in data:
          for key in data[layer]:
            g[layer][key] += data[layer][key]

      for layer in g:
        for key in g[layer]:
          g[layer][key] = g[layer][key] / size

      updates, opt_state = optimizer.update(g, opt_state, params)
      params = optax.apply_updates(params, updates)

      for i in range(1, size):
        comm.send(params, dest=i)

      if (step % config.test.test_every) == 0:
        rng_key, val_key = jax.random.split(rng_key)
        metrics = jitted_eval(params, val_key)
        print(f'Valid[{step}]: {metrics}')
    else:
      comm.send(g, dest=0)
      params = comm.recv(source=0)

    step += 1

  print('Done')


if __name__ == '__main__':
  app.run(main)
