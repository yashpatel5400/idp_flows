# idp_flows
Normalizing flows-based generative modelling for IDPs

Credit for base to: https://github.com/deepmind/idp_flows

## Great Lakes Instructions

```bash
module load cuda/11.5.1
python -m pip install --upgrade jaxlib==0.3.10+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install --upgrade "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install dm-haiku
python -m pip install ml-collections
python -m pip install optax
python -m pip install distrax
PYTHONPATH=/home/[uniqname] python experiments/train.py
```