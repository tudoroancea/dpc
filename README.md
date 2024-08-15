# dpc

## setup
We provide below a setup example using [`uv`](https://github.com/astral-sh/uv), but you should be able to do everything with plain `venv` and `pip`.

```bash
git clone https://github.com/tudoroancea/dpc
cd dpc
uv venv
. .venv/bin/activate
uv pip install -e '.[dev]'
```

If you are on linux, also do the following:

```bash
uv pip install -e '.[linux]' 
```

Finally, install the nightly wheels of `casadi` from https://github.com/tudoroancea/casadi/releases/tag/nightly-main.
Once version 3.6.7 is released, we should be able to include it directly in the project dependencies in `pyproject.toml`.

### casadi setup:

To be able to use [fatrop](https://github.com/meco-group/fatrop), install the nightly wheels from https://github.com/casadi/casadi/releases/tag/nightly-plugin_madnlp2.

## discoveries

- The cost is very big because we have very extreme initial conditions, that don't often appear in the closed loop experiments we did. 
  We can keep these but maybe do several training phases:

  1. "_Pretraining_" on this dataset with a lot of different initial conditions
  2. "_Finetune_" on a dataset concentrated around the reference (gaussian error model) that should be more representative of the conditions appearing in closed-loop.

  The tricky part will be to make sure the model doesn't forget too much :sweat_smile:

- why is delta so noisy?  can we add some regularization? that is not in the original NMPC problem?

- cost for NMPC with initial pose $(0,0.5,\pi/2-0.5)$ is ≈4500. => How can we optimize for the edge cases without? should we scale the cost by the initial vel error?


## coincidence?
Things that happen at almost every training, no matter the net dimensions, no matter even the dataset.
1. the validation spikes at the very beginning and then converges to the train loss.
  ![](doc/initial_validation_spike.png)
  ![](doc/initial_validation_spike_2.png)
2. at some point, the loss starts oscillating a lot, even with `lr=1e-4`
  ![](doc/oscillation.png)
  ![](doc/oscillation_2.png)


## dataset format

CSV file with the following format:
```
X,Y,phi,v,X_ref_0,Y_ref_0,phi_ref_0,v_ref_0,...,X_ref_40,Y_ref_40,phi_ref_40,v_ref_40,T_mpc_0,delta_mpc_0,...,T_mpc_39,delta_mpc_39,cost_mpc
```
