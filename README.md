# Bridging dimensions: confident reachability for high-dimensional controllers
Here are the codes for the paper "Bridging dimensions: confident reachability for high-dimensional controllers". This repository contains the following contents:
1. The low-dimensional controller training code with naive-splitting
2. Calculating the action-based and trajectory-based conformal prediction discrepancy.
3. Revised POLAR code to apply reachability analysis with different discrepancies.
4. Well-trained low-dimensional controllers saved as a text file.
5. Three environments contain the continuous control action space and images as input 

## Prerequisites
```python
pip install -r requirements.txt
```
Note that we use gym 0.21.0 in the cart pole case and for others, we apply gym 0.22.0. 

Regarding the reachability analysis, download our VirtualBox with all dependencies. Here is the link: https://www.dropbox.com/scl/fi/ki122ofypp1x0tmq5nunn/ReachNNStar-test-2.ova?rlkey=a0l7raqkvpa87jaw98ygr1mme&st=xstj3k0l&dl=0 

Otherwise, you can also download the dependency by yourself. System Requirements: Ubuntu 18.04, MATLAB 2016a or later
Install dependencies through apt-get install
```python
sudo apt-get install m4 libgmp3-dev libmpfr-dev libmpfr-doc libgsl-dev gsl-bin bison flex gnuplot-x11 libglpk-dev gcc-8 g++-8 libopenmpi-dev libpthread-stubs0-dev
```
Then download the Flow*
```python
git clone https://github.com/chenxin415/flowstar.git
```
Compile flow star:
```python
cd flowstar/flowstar-toolbox
make
```
Download the POLAR Toolbox:
```python
git clone https://github.com/chenxin415/flowstar.git
```
Compile POLAR:
```python
cd POLAR
make
```


## Get ground truth data from HDC and training data for LDC
For state-image one-to-one matched training datasets, we collect the images from env. render() and store the corresponding states in the env.env.state in each step function. Moreover, we add the zero-mean Gaussian noise onto the state information for the noise-mapping training data set. For example, the arguments are number of data, steps, initial state1 begin, initial state1 end, initial state2 begin, initial state2 end.   

```bash
python MC_LDC_traing/MC_training_data.py 10000 60 -0.6 -0.4 0.01 0.05
```

## Train the LDC with knowledge-distillation
After getting the training data, we train a series of LDCs by considering the conformal prediction value and MSE. Still take the Mountain car as an example, given the "name_training_data.npy", beginning initial state1, end initial state1, beginning initial state2, end initial state2, it will train multiple LDCs that are saved as txt files.

```bash
python MC_LDC_traing/train_test_LDC.py "training_data_10000.npy" -0.6 -0.4 0.01 0.05
```

## Calculate the action- and trajectory-based discrepancy
We apply the conformal prediction to calculate the statistical bound for actions and trajectories. Given the LDC, HDC, initial set, and sample points, we can get the conformal bound by simply picking up the quantile. For instance,

```bash
python MC_CP/new_action_based_MC_LDC.py LDC1.txt trained_HDC_cnn_model.h5 60 -0.6 -0.4 0.01 0.05
```

## Reachability analysis in POLAR
Once we trained all the LDCs and the statistical discrepancy, we ran the reachability analysis on these LDCs with these bounds. Take the Mountain car as an example,
```python
make mountain_car && ./mountain_car 0.01 60 4 6 1
```
Where 0.01 is the width of the initial set, 60 is the total steps that need to be verified, 4 is the order of Bernstein Polynomial, 6 is the order of the Taylor Model.
One safe and unsafe verification results are displayed below.
<img src="/MC_after_POLAR/2Successful_verificaiton_plot.png" alt="alt text" width="300" height="200"/>
<img src="/MC_after_POLAR/2Failed_verificaiton_plot.png" alt="alt text" width="300" height="200"/>


## Verification results
Considering the results from the POLAR and the ground truth from the first step, we can get the confusion matrix for true positive rate, false negative rate, and precision to check our theory and compare different methods.

## Contribution
[Yuang Geng](https://github.com/yuanggeng), Jake Brandon Baldauf, [Souradeep Dutta](https://github.com/souradeep-111), [Chao Huang](https://github.com/ChaoHuang2018), and [Ivan Ruchkin](https://github.com/bisc)
