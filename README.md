# adapCNV
adapCNV: A tool for detecting copy number variations by dynamically adjusting parameters

## install
Uncompress the installation zip:

```bash
$ cd /my/install/dir/
$ unzip /path/to/adapCNV.zip
```
**Requirements**

You'll need to install the following Python packages in order to run the code:

```bash
numpy
pysam
sys
pandas
numba
subprocess
imblearn
sklearn.decomposition 
scipy.optimize
joblib
os
gzip
re
pyod.models.iforest

``
## Usage
```bash
$ python adapCNV-pro.py tumor.bam reference output max_binsize
```
## Maintainers
@Y-J124
