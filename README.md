# NODE with non-local latent encoding for prediction of trajectories of dynamical systems
-----------------------------------------------------------


## Intro to a poblem
The forecasting of the dynamic systems is a long known problem having a variety of real-life
applications in many areas. Essentially, every time series data could be viewed as discrete samples
of some underlying dynamic system trajectories, which makes this problem even more common. Different formulations of the problem could be approached, but the one
that was selected is as follows:
k=1,N

Consider a set of $`N`$ discrete sequences of length $`n+1`$, $`\langle y^k_i \rangle _{i = \overline{0, n}} ^{k = \overline{1, N}} `$,
such that the elements of that sequence are samples of certain functions $`y^k(t)`$, sampled at times $`t_i`$.
We want to be able to approximate these samples and predict the evolution of the system into the future.

---------------------------------------------------------

## Requirements
To install this dependencies run 

```
pip install -r requirements.txt
```
------------------------------------------------------------
## Quick start and results 
 
