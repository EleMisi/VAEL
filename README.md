# VAEL

Codebase for [_VAEL: Bridging Variational Autoencoders and Probabilistic Logic Programming_](https://papers.nips.cc/paper_files/paper/2022/hash/1e38b2a0b77541b14a3315c99697b835-Abstract-Conference.html) published at NeurIPS 2022.

If you use this codebase, please cite:

```
@inproceedings{NEURIPS2022_1e38b2a0,
 author = {Misino, Eleonora and Marra, Giuseppe and Sansone, Emanuele},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {4667--4679},
 publisher = {Curran Associates, Inc.},
 title = {VAEL: Bridging Variational Autoencoders and Probabilistic Logic Programming},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/1e38b2a0b77541b14a3315c99697b835-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}
```



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
        
## Corresponding Author

* [Eleonora Misino](https://github.com/EleMisi) - eleonora.misino2@unibo.it
* [Giuseppe Marra](https://github.com/GiuseppeMarra) - giuseppe.marra@kuleuven.be
* [Emanuele Sansone](https://github.com/emsansone) - emanuele.sansone@kuleuven.be

