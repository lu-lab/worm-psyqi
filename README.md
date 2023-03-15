# worm-psyqi

WormPsyQi is a semi-automated pipeline for high-throughput analysis of synapses in the nematode C. elegans. Using fluorescent protein-based techniques, WormPsyQi can quantitate multiple features of synapses, including number, size, intensity, and spatial distribution. We provide a pure python-based GUI.

## Installation
Clone this repository and use the provided [yml file](./conda-environments/WORMPSYQI.yml) to setup an Anaconda environment.
```consol
$ conda env create -f ./conda-environments/WORMPSYQI.yml
```
To check if the environment is installed successfully:
```consol
$ conda activate wormpsyqi
```
You will see your shell has (wormpsyqi) at the start as below.
```consol
(wormpsyqi) $ 
```

## Run the pipeline
Please see [Running_guide.pdf](./Running_guide.pdf)