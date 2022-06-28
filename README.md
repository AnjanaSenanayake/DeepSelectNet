# DeepSelecNet
DeepSelecNet is an improved version of the original (https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02511-y) 1D ResNet based model to classify Oxford Nanopore raw electrical signals as target or non-target for Read-Until sequence enrichment or depletion. DeepSelecNet provides enhanced model performances.

### Abstract

> Oxford Nanopore Technologies (ONT) introduced MinION, the first portable commercial sequencer of its kind in 2014. The MinION, being a small handheld device, was able to revolutionize the genomics community by means of its superiority in portability. MinION enables access to real-time current signals when a DNA traverses through the pore. ONTs capability to produce long reads is an added advantage in sequencing to have better genome assemblies. But having to sequence unwanted regions of a genome or unwanted species in a given genome pool is one of the main drawbacks of long reads since it would cost a lot of pore time, reducing the efficiency of sequencing. For example, a DNA strand of one million bases (1Mb) in length on average would take about one hour of sequencing. Therefore, improving the speed of targeted sequencing/selective sequencing in ONT platforms is a widely discussed topic. An efficient and real-time analysis of the current signals from the nanopore would enable performing selective sequencing on a DNA sample with the help of ReadUntil API from ONT which can reverse the current flow in a nanopore to eject a sequencing DNA strand and start sequencing for a new strand. However, existing methods on the current signal to targeted reference genome comparisons are timewise costly and hence would give away the real-time capability of sequencing. Therefore, potential improvements to the performance of selective sequencing in many areas are yet to be discovered. Our approach of deep neural network-based selective sequencing would study possibilities to explore the improved methods of selective sequencing with better accuracy and speed compared to state-of-the art methodologies. Recently in basecalling, there have been breakthrough improvements made with deep neural networks over then existing deterministic approaches. Similarly in selective sequencing, use of customized deep neural network architecture would potentially be beneficial in improving accuracy, speed and also as a sample independent methodology. 

## Installation
> Create a python virtual environment in Python3.8

#### On Linux:
* Open a terminal in the root directory of the code repository.
* Create a python3 virtual environment named `deepselecenv`

```
python3 -m venv deepselecenv
```

* Use the following command to activate the virtual environment created.

```
source deepselecenv/bin/activate
```

* Install required packages in the virtual enviroment. 

```
pip install -r requirements.txt
```

* [Optional] To Leave the environment when not in use.

```
deactivate
```

## Scripts

#### 1. preprocessor.py
Preprocess the fast5 files into numpy dumps so that they can be used for training
- Options
  * f5   - Path to fast5 directory
  * c    - Read signal cutoff value (default=1500)
  * sz   - Read signal sample size (default=1000)
  * sco  - Subsampling coefficient/Number of random samplings from a single read (default=1)
  * b    - Number of fast5 read samples should be in a numpy dump (default=1000)
  * pico - Is enabled pico conversion (default=True)
  * lb   - Class label of the preprocessing dataset (default=1) | 1 -> positive class | 0 -> negative class
  * mad  - Median absolute deviation value for data normalization (default=3)
  * rep  - Is repeatedly normalized or not (default=False)
  * o    - Numpy dump output path
```
python preprocessor.py -f5 <fast5_dir> -c 1500 -lb 1 -mad 5 -o <output_dir>
```

#### 2. trainer.py
Train the model for given dataset using dumped numpy arrays
- Options
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
python trainer.py -d <npy_dump_dir> -c ResNet -lf bc -s 0.7 -k 5 -e 200 -o <output_dir>
```

#### 3. inference.py
Predict the class of unseen fast5 reads with trained model
- Options
  * model - Path to trained model directory
  * f5    - Path to fast5 directory required to predict
  * b     - The batch size (default=1)
  * c     - The read signal cutoff value (default=1500)
  * sz    - Read signal sample size (default=1000)
  * sco   - Subsampling coefficient/Number of random samplings from a single read (default=1)
  * pico  - Is enabled pico conversion (default=True)
  * lb    - Class label of the preprocessing dataset (default=1) | 1 -> positive class | 0 -> negative class
  * mad   - Median absolute deviation value for data normalization (default=3)
  * rep   - Is repeatedly normalized or not (default=False)
  * o     - Predictions output path
```
python inference.py -model <saved_model_dir> -f5 <fast5_dir> -lb 1 -mad 5 -o <output_dir>
```

### [Support Scripts](support/README.md#support-scripts)
