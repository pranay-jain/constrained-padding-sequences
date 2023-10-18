# Near-Optimal Constrained Padding for Object Retrievals with Dependencies

This repo contains the datasets for the paper *"Near-Optimal Constrained Padding for Object Retrievals with Dependencies"* -- currently under review at USENIX 2024. The `data/` directory consists of three directories, corresponding to the datasets used in the study. 

### Wikipedia Dataset

This dataset consists of a set of 2,805 wikipedia pages and their sizes scraped from the web. It also contains the hyperlinked edges between these pages and sequences constructed by random walks through the hyperlinks. It also consists of page views from these wikipedia pages.

It also consists of a "past" wikipedia dataset, which models the same pages and hyperlinked edges between them from a past date (constructed using the Wayback API). This data is different from the current data, since in the past some pages may not exist and some hyperlinks may be different. Using this, researchers can test the evolution of the graph of page hyperlinks on Wikipedia.

### Autocomplete Dataset

This dataset consists of a list of 899 words. For each word, a user is modeled to have queried it one letter at a time in the Google search bar and receives autocomplete suggestions. The page sizes of the autocomplete suggestions are provided in the dataset, as well as the number of search results for the words, providing a proxy for a probability distribution.

### Linode Dataset

This dataset was created by crawling the Linode documentation website. It models the web surfing behavior of a user navigating to a page of interest on a website consisting of a large number of interconnected webpages. This dataset consists of vertices -- corresponding to pages -- along with their page sizes. The page size in this case is the combined size of all artifacts on that page (including images). It also consists of edges and sequences. In this dataset, the sequences from a tree structure in that all sequences start at the index page and form a path to another page.


