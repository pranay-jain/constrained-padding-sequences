# Near-Optimal Constrained Padding for Object Retrievals with Dependencies

This repo contains the datasets and code for our paper:

__"Near-Optimal Constrained Padding for Object Retrievals with Dependencies"__  
[Pranay Jain](https://www.linkedin.com/in/pranayjain1), [Andrew C. Reed](https://andrewreed.io), and [Michael K. Reiter](https://reitermk.github.io)  
[33rd USENIX Security Symposium](https://www.usenix.org/conference/usenixsecurity24), August 2024

## Installation

All of our executable code is written in Python, either in .py scripts or in Jupyter notebooks. To set up the required packages, we recommend using a Python virtual environment, such as `conda`, `virtualenv` or `venv`. Once you have activated your virtual environment of choice, run the command:
```
pip install -r requirements.txt
```

For `python3` environments, replace `pip` with `pip3` if needed. This will install all the packages required by the code in this repository.

#### Gurobi

Our code uses the Gurobi Optimizer. If you do not already have access to Gurobi, our recommendation is to obtain an _Academic Named-User License_. Instructions for how to obtain a Gurobi academic license can be found at:

[Gurobi Academic License Program](https://www.gurobi.com/academia/academic-program-and-licenses)

## Organization

This repo is organized as follows:
1. **data/** - Our three datasets are located in subdirectories here.
2. **experiments/** - The Jupyter notebooks that produce our Figures (from the paper) are here.
3. **repo root dir** - The implementation of our algorithm, as well as our implementations of the algorithms to which we compare, are located in the root directory of this repo.

## Running the Experiments

We provide Jupyter notebooks that run each of our primary experiments from the paper. To run them, first start an instance of Jupyter Notebook and then navigate to the `experiments/` directory and open the .ipynb file named after the figure that you'd like to run. Each notebook is designed to be run as-is, i.e., `Kernel > Restart & Run All`.

## Datasets

Descriptions of our datasets can be found in Section 5.1 of our paper.

The function `load_dataset` located in `load_dataset.py` loads each of our datasets. For the purposes of running our experiments, everything is automated within the Jupyter notebooks (described above). Information about the function's input arguments and its output can be found in `load_dataset.py`.

#### I THINK WE STOP HERE

#### Pranay
I think this needs to go as documentation in load_dataset.py:

The input arguments for `load_dataset` are as follows:
1. `dataset`: a string that should be one of {`autocomplete`, `linode_from_index`, `wikipedia`}
2. `cap_sequences`: a Boolean that indicates if the sequences should be truncated when loaded
3. `cap_length`: an integer; if `cap_sequences` is set to `True` then sequences are truncated to this length

For our experiments, we left `cap_sequences` to `False`, i.e., we did not leverage that feature of `load_dataset`.

`load_dataset` returns the following output:
1. `vertices` - a dictionary where the key is the object's name and the value is its size (in bytes)
2. `vertices_subset` -
3. `sequences` - 
4. `prefix_closed_sequences` - 
5. `max_length` - 
6. `edges` - 
7. `Q` - 

#### I THINK WE DELETE THE REST OF THIS

## Running the model

#### Padding For Sequences (PFS)

This is the proposed algorithm for producing near-optimal padding scheme for sequences. To run this algorithm, execute the `pfs.py` code. Sample execution:

```
python main.py -d wikipedia -c 1.05
```

#### Parameters

1. [**Required**] Dataset (use flag `-d` or `--dataset`) : This selects the dataset to run the model against. The choices for this parameter are `{'wikipedia', 'linode_from_index', 'autocomplete', 'synthetic'}`. Each of these datasets are explained below. The datasets are stored inside the `data` directory.

2. [**Required**] Padding factor `c` (use flag `-c` or `--pad_factor`): This value sets the maximum padding factor allowed to the PFS or PWOD algorithms. The padding factor for an object of size `x` is defined as `pad(x) / x`. We recommend a value for `c` in `{1.05, 1.25, 1.50, 2.00}`.

3. Stride count `k` (use flag `-k` or `--stride`):

4. Prefix closed 


### Wikipedia Dataset

This dataset consists of a set of 2,805 wikipedia pages and their sizes scraped from the web. It also contains the hyperlinked edges between these pages and sequences constructed by random walks through the hyperlinks. It also consists of page views from these wikipedia pages.

It also consists of a "past" wikipedia dataset, which models the same pages and hyperlinked edges between them from a past date (constructed using the Wayback API). This data is different from the current data, since in the past some pages may not exist and some hyperlinks may be different. Using this, researchers can test the evolution of the graph of page hyperlinks on Wikipedia.

### Autocomplete Dataset

This dataset consists of a list of 899 words. For each word, a user is modeled to have queried it one letter at a time in the Google search bar and receives autocomplete suggestions. The page sizes of the autocomplete suggestions are provided in the dataset, as well as the number of search results for the words, providing a proxy for a probability distribution.

### Linode Dataset

This dataset was created by crawling the Linode documentation website. It models the web surfing behavior of a user navigating to a page of interest on a website consisting of a large number of interconnected webpages. This dataset consists of vertices -- corresponding to pages -- along with their page sizes. The page size in this case is the combined size of all artifacts on that page (including images). It also consists of edges and sequences. In this dataset, the sequences from a tree structure in that all sequences start at the index page and form a path to another page.


