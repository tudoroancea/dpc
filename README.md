# dpc

- The cost is very big because we have very extreme initial conditions, that don't often appear in the closed loop experiments we did. 
  We can keep these but maybe do several training phases:

  1. "_Pretraining_" on this dataset with a lot of different initial conditions
  2. "_Finetune_" on a dataset concentrated around the reference (gaussian error model) that should be more representative of the conditions appearing in closed-loop.

  The tricky part will be to make sure the model doesn't forget too much :sweat_smile:


