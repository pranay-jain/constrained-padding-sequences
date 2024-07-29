# Near-Optimal Constrained Padding for Object Retrievals with Dependencies

This repo contains the datasets and code for our paper:

__"Near-Optimal Constrained Padding for Object Retrievals with Dependencies"__  
[Pranay Jain](https://www.linkedin.com/in/pranayjain1), [Andrew C. Reed](https://andrewreed.io), and [Michael K. Reiter](https://reitermk.github.io)  
[33rd USENIX Security Symposium](https://www.usenix.org/conference/usenixsecurity24), August 2024

[![DOI](https://zenodo.org/badge/706495481.svg)](https://zenodo.org/doi/10.5281/zenodo.13119686)


## Installation

All of our executable code is written in Python, either in .py scripts or in Jupyter notebooks. To set up the required packages, we recommend using a Python virtual environment, such as `conda`, `virtualenv` or `venv`. Once you have activated your virtual environment of choice, run the command:
```
pip install -r requirements.txt
```

For `python3` environments, replace `pip` with `pip3` if needed. This will install all the packages required by the code in this repository.

#### Gurobi

Our code uses the Gurobi Optimizer. If you do not already have access to Gurobi, our recommendation is to obtain an _Academic Named-User License_. Instructions for how to obtain a Gurobi academic license can be found at:

[Gurobi Academic License Program](https://www.gurobi.com/academia/academic-program-and-licenses)

**Note on Gurobi License Installation:**

The final step of setting up the Gurobi license requires you to run the command `grbgetkey` with the serial number provided to you by Gurobi. This command **must** be executed while your computer is connected to your academic network.

If your computer is not connected to your academic network, then using a VPN connected to the academic network will work. However, sometimes VPNs are only configured for split-tunneling, i.e., only traffic destined for the academic network itself will be routed through the VPN. In those instances, if you have SSH access to a server located on the academic network, then here is a potential workaround:
1. Connect to your academic network using your VPN.
2. In a terminal window, use a program such as `sshuttle` to route **all** internet traffic through SSH. This can be done using the following command: `~/anaconda3/bin/sshuttle -r username@servername 0/0 -x servername`. In the preceding command, adjust the path to `sshuttle` according to your own development environment, and replace `username` and `servername` with your own credentials for the SSH server that you will be using.
3. In a second terminal window, run the command `grbgetkey` with the serial number provided to you by Gurobi.
4. Once the Gurobi license has been activated, you can close the `sshuttle` connection using `CTRL-C` in the first terminal window, and then disconnect from the VPN.

## Organization

This repo is organized as follows:
1. **data/** - Our three datasets are located in subdirectories here.
2. **experiments/** - The Jupyter notebooks that produce our Figures (from the paper) are here.
3. **repo root dir** - The implementation of our algorithm, as well as our implementations of the algorithms to which we compare, are located in the root directory of this repo.

## Running the Experiments

We provide Jupyter notebooks that run each of our primary experiments from the paper. To run them, first start an instance of Jupyter Notebook and then navigate to the `experiments/` directory and open the .ipynb file named after the figure that you'd like to run. Each notebook is designed to be run as-is, i.e., `Kernel > Restart & Run All`.

## Datasets

Descriptions of our datasets can be found in Section 5.1 of our paper.

The function `load_dataset` located in `load_dataset.py` loads each of our datasets. For the purposes of running our experiments, everything is automated within the Jupyter notebooks (described above).

## Running Padding For Sequences (PFS)

This is the proposed algorithm for producing near-optimal padding scheme for sequences. The `experiments` directory shows how the model (and baselines) can be executed in notebooks. This section describes how to execute the PFS algorithm as a python executable. To run this algorithm, execute the `main.py` code. Sample execution:

```
python main.py -d wikipedia -c 1.05
```

#### Parameters

1. [**Required**] Dataset (use flag `-d` or `--dataset`) : This selects the dataset to run the model against. The choices for this parameter are `{'wikipedia', 'linode_from_index', 'autocomplete', 'synthetic'}`. Each of these datasets are explained below. The datasets are stored inside the `data` directory.

2. [**Required**] Padding factor `c` (use flag `-c` or `--pad_factor`): This value sets the maximum padding factor allowed to the PFS or PWOD algorithms. The padding factor for an object of size `x` is defined as `pad(x) / x`. We recommend a value for `c` in `{1.05, 1.25, 1.50, 2.00}`.

3. [Optional] Stride count `k` (use flag `-k` or `--stride`):

4. [Optional] Prefix closed (use flag `--prefix-closed`): This is a boolean argument which selects if the prefix-closed set of sequences is used as input to the model. The definition for prefix closure is provided in the paper.
