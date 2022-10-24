# DeepSelecNet
DeepSelecNet is an improved 1D ResNet based model to classify Oxford Nanopore raw electrical signals as target or non-target for Read-Until sequence enrichment or depletion. DeepSelecNet provides enhanced model performances.

### Abstract

> Nanopore sequencing allows selective sequencing, the ability to programmatically reject unwanted reads in a sample. Selective sequencing has many present and future applications in genomics research and the classification of species from a pool of species is an example. Existing methods for selective sequencing for species classification are still immature and the accuracy highly varies depending on the datasets. For the five datasets we tested, the accuracy of existing methods varied in the range of ∼77%-97% (average accuracy <89%). Here we present DeepSelectNet, an accurate deep-learning-based method that can directly classify nanopore current signals belonging to a particular species. DeepSelectNet utilizes novel data preprocessing techniques and improved neural network architecture for regularization. 

## Installation

### Prerequisities
* Python 3.5 >= version <= 3.8
* Python venv

#### Steps(On Linux)
1) Open a terminal in the root directory of the code repository.
2) Create a python3 virtual environment named `deepselectenv`

```
python3 -m venv deepselectenv
```

3) Use the following command to activate the virtual environment created.

```
source deepselectenv/bin/activate
```

4) Install required packages in the virtual enviroment. 

```
pip install -r requirements.txt
```

5) [Optional] To Lave the environment when not in use.

```
deactivate
```

## Scripts

#### 1. Preprocessor
Preprocess the slow5 files into numpy dumps so that they can be used for training
- Args
  * pos_s5   - Path to positive slow5 file
  * neg_s5   - Path to negative slow5 file
  * c    - Read signal cutoff value (default=1500)
  * sz   - Read signal sample size (default=3000)
  * sco  - Subsampling coefficient/Number of random samplings from a single read (default=4)
  * b    - Number of slow5 read samples should be in a numpy dump (default=20000)
  * pico - Is enabled pico conversion (default=True)
  * mad  - Median absolute deviation value for data normalization (default=3)
  * rep  - Is repeatedly normalized or not (default=False)
  * o    - Numpy dump output path
```
python scripts/preprocessor.py -pos_s5 <pos_slow5> -neg_s5 <neg_slow5> -b 20000 -c 1500 sco 4 -mad 3 -o <output_dir>
```
Note:
- Num. of reads from source(pos_s5, neg_s5) should be enough to generate a balanced dataset with Num. of reads equal to batch size(b).
- Read lengths in source reads should be larger than cutoff value(c) + sample size(sz).
- [support scripts](support) may come useful in these manipulations.

#### 2. Trainer
Train the model for given dataset using dumped numpy arrays
- Args
  * d   - Path to numpy dump directory
  * c   - Classifier architecture -> FCN|ResNet|InceptionNet|TransformerNet (default=ResNet)
  * lf  - Loss function of the model -> bc|cc|scc (default=bc)
  * s   - The split ratio between train and validation (default=0.75)
  * occ - Perform one class classification or not (default=False)
  * oh  - Is one hot encoded labels or not (default=False)
  * k   - Number cross validation folds (default=10)
  * e   - Number epochs (default=10)
  * b   - Batch size (default=1000)
  * o   - Trained model output path
```
python scripts/trainer.py -d <npy_dump_dir> -s 0.7 -k 5 -e 200 -o <output_dir>
```

#### 3. Inference
Predict the class of unseen slow5 reads with trained model
- Args
  * model - Path to trained model directory
  * s5    - Path to slow5 directory required to predict
  * b     - The batch size (default=1)
  * c     - The read signal cutoff value (default=1500)
  * sz    - Read signal sample size (default=3000)
  * sco   - Subsampling coefficient/Number of random samplings from a single read (default=1)
  * pico  - Is enabled pico conversion (default=True)
  * lb    - Class label of the preprocessing dataset (default=1) | 1 -> positive class | 0 -> negative class
  * mad   - Median absolute deviation value for data normalization (default=3)
  * rep   - Is repeatedly normalized or not (default=False)
  * o     - Predictions output path
```
python scripts/inference.py -model <saved_model_dir> -s5 <slow5_dir> -lb 1 -mad 3 -o <output_dir>
```

### [Support Scripts](support)
