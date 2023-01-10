# Experimental design: comparison of Statistical Reliability methods for Neural Networks

## 1. Goal 

The goal is to benchmark different methods of Statistical Reliability Engineering in the context of Neural Networks classifiers.

The methods to compare include:

- [x] Crude Monte Carlo (MC)
- [x] Multiple Importance Splitting (MLS)
- [x] Hamiltonian within Sequential Monte Carlo (H_SMC)
- [x] Metropolis Adjusted Langevin Algorithm within Sequential Monte Carlo (MALA_SMC)
- [ ] First Order Reliability Method (FORM)
- [ ] Importance Sampling and Cross-Entropy methods (IS/CE)
- [ ] Line Sampling methods (LS)

## 2. Scope of experiments

We evaluate each estimation method on 3 test cases:

- [ ] A linear toy model under Gaussian perturbations, see [`exp_1`](./exp_1) 
- [ ] Models trained on MNIST dataset, see [`exp_2`](./exp_2) 
- [ ] Models trained on ImageNet dataset, see [`exp_3`](./exp_3) 

## 3. Rare Event Threshold

We study Rare Events in the case of Neural Networks misclassifications. However 'Rare' is an ambiguous term. Therefore on the toy model and and the MNIST models we study the probability level threshold such that a Crude Monte Carlo simulation become less efficient than Rare Event Simulation algorithms.
This corresponds to the experiments scripts [`exp_1_high_low_prob.py`](./exp_1_high_to_low_prob.py) and .
