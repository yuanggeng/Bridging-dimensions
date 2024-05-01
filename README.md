# Bridging dimensions: confident reachability for high-dimensional controllers
Here are the codes for the paper "Bridging dimensions: confident reachability for high-dimensional controllers". This repository contains the following contents:
1. The low-dimensional controller training code with naive-splitting
2. Calculating the action-based and trajectory-based conformal prediction discrepancy.
3. Rivsed POLAR code to apply reachability analysis with different discrepancies.
4. Well-trained low-dimensional controllers saved as a text file.
5. Three environments contain the continuous control action space and images as input 

## Prerequisites
```python
pip install -r requirements.txt
```
Note that we use gym 0.21.0 in the cart pole case and for others, we apply gym 0.22.0. 

Regarding the reachability analysis, download the VirtualBox and run the code inside. Here is the link: https://www.dropbox.com/scl/fi/ki122ofypp1x0tmq5nunn/ReachNNStar-test-2.ova?rlkey=a0l7raqkvpa87jaw98ygr1mme&st=xstj3k0l&dl=0 

## Get ground truth data from HDC and training data for LDC
For state-image one-to-one matched training datasets, we collect the images from env. render() and store the corresponding states in the env.env.state in each step function. Moreover, we add the zero-mean Gaussian noise onto the state information for the noise-mapping training data set. For example, the arguments are number of data, steps, initial state1 begin, initial state1 end, initial state2 begin, initial state2 end.   

```bash
python MC_LDC_traing/MC_training_data.py 10000 60 -0.6 -0.4 0.01 0.05
```

## Train the LDC with knowledge-distillation


## Calculate the action- and trajectory-based discrepancy


## Reachability analysis in POLAR


## Miscellaneous test 
1. Train the LDC first and gather the training data (also contains the ground truth for safety verification) in the three py files, “Mountain_car_simulaiton.py, Train_HDC.py, train_test_LDC.py”. For the first training, we only aim to decrease the MSE as much as we can. If there is a high overapproximation error in the verification before inflation, we switch to the verification-oriented KD method to retrain the LDC by balancing the MSE and Lipschitz constants. “https://github.com/JmfanBU/ReachNNStar/tree/master/VF_retraining”
2. After getting the action-based discrepancy, we try to inflate the interval in the Taylor model in POLAR. Concerning the trajectory-based discrepancy, we extracted the reachable tube (a sequence of polygons) and inflate it with the trajectory-based discrepancy. With respect to the action-based discrepancy, we inflate the interval by action-based discrepancy from the Taylor model. This process involves inserting the CP table, picking up the correct CP value in different initial set or state space, and so on. 
