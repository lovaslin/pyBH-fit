# pyBH-fit

This repository rasemble all the codes used to perform the fit of the BumpHunter test statistics distribution presented in the article :
"Fit of the BumpHunter test statistic distribution and global p-valus evaluation".

The results were obtained using pyBumpHunter v0.4.0.

[Link](https://github.com/scikit-hep/pyBumpHunter) to pyBumpHunter repository  
Link to the paper [not yet]

## Instruction to run the tests

* Create a new python environement
```bash
python3 -m venv .env
source .env/bin/activate
```

* Install the required packages
```bash
pip install -r requirement.txt
```

* Run the test you want  
The comand line comes with a number of arguments that are used to tune some parameters.  
The available arguments are :  
  - Nb : the number of bins in the scanned distribution (set to 20 by default)
  - loc_path : Path to the directory where the final plots should be stored
  - tem_path : Path to the directory where the teporary files generated when scanning the PEs are stored
  - width : Width of the window used by BumpHunter to scan the distribution (default to 1)
  - bkg : Define the shapes of the background in the scanne distribution, can be either 'exp' or 'lin' (default to 'exp')
  - Nbkg : Define the number of background event generated in the scanned histograms (default to 100k)
 
 Example of command line used to run the test :
 ```bash
 python3 fit_test.py --Nb 20 --bkg exp --Nbkg 1000 --width 1
 ```
 
 Optionnally the script `run_all.sh` can be used to run automatically all the tests to reproduce the results of the paper.
