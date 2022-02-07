# VAEL

Codebase for _Bridging Variational Autoencoders and Probabilistic Logic Programming_.



## Prerequisites

* Python 3.7
* Dependencies:
  ```sh
  pip install -r requirements.txt
  ```

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

