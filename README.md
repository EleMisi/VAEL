# VAEL

Codebase for _Bridging Variational Autoencoders and Probabilistic Logic Programming_.
We kindly ask you to cite our work if you use this codebase:

    @misc{https://doi.org/10.48550/arxiv.2202.04178,
      doi = {10.48550/ARXIV.2202.04178},
      url = {https://arxiv.org/abs/2202.04178},
      author = {Misino, Eleonora and Marra, Giuseppe and Sansone, Emanuele},
      keywords = {Programming Languages (cs.PL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {VAEL: Bridging Variational Autoencoders and Probabilistic Logic Programming},
      publisher = {arXiv},
      year = {2022},
      copyright = {arXiv.org perpetual, non-exclusive license}
    }




## Prerequisites

* Python >=3.7 
* Dependencies:
  ```sh
  pip install -r requirements.txt
  ```
  _Note_: if something goes wrong with PySDD, try `pip install -vvv --upgrade --force-reinstall --no-binary :all: --no-deps pysdd`

## Usage

1. Clone the repo
   ```sh
   git clone https://github.com/EleMisi/VAEL.git
   ```
2. Install the dependencies
   ```sh
   pip install -r requirements.txt
   ```
3. Set the experiment(s) configuration in file _config.py_
   
4. Run the experiment(s)
   ```sh
   python run_VAEL.py
   ```
   Use flag `--task mnist` to run _2digit MNIST_ experiment(s), and `--task mario` to run _Mario_ experiment(s).

## Results
The results are stored in the folder ./<exp_folder>/<exp_class>/ specified in _run_VAEL.py_.  

In particular:
   *  the resulting metrics for each tested configuration are reported in _exp_class.csv_
   * each subfolder refers to a specific configuration and contains
      * the model checkpoint
      * the learning curves 
      * some samples of image reconstruction and generation
## Author

* [Eleonora Misino](https://github.com/EleMisi) - eleonora.misino2@unibo.it
* [Giuseppe Marra](https://github.com/GiuseppeMarra) - giuseppe.marra@kuleuven.be
* [Emanuele Sansone](https://github.com/emsansone) - emanuele.sansone@kuleuven.be

