# dpc

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