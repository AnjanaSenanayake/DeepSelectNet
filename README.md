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

## Support Scripts

#### 1. event_sort.py
Sorts events in a tsv file in ascending order
- Options
  * tsv - Path to tsv file
  * b - The number of events to be sorted for one output
```
python event_sort.py -tsv <tsv_file.tsv> -b 1000
```
#### 2. export_read_id.py
Sorts events in a tsv file in ascending order
- Options
  * f5 - Path to fast5 file
  * o - Path to output file
```
python export_read_id.py -f5 <fast5_file> -o <output_file.txt>
```
#### 3. fast5_filter.py
Filter single_fast5 files in a directory with read length or compared to a list
- Options
  * f5 - Path to fast5 directory
  * c - The read signal cutoff value (default=1500)
  * sz - Read signal sample size (default=3000)
  * type - Filter type | Length, List (default=Length)
  * list - List of fast5 names,if the filter type is list
```
python fast5_filter.py -f5 <fast5_dir> -c 1500 -sz 3000 -type Length
```
```
python fast5_filter.py -f5 <fast5_dir> -c 1500 -sz 3000 -type List -list <fast5_list.txt>
```
#### 4. fast5_plotter.py
Plots the raw signal in fast5 files
- Options
  * f5 - Path to fast5 directory
  * mad - Median absolute deviation value for data normalization (default=3)
  * rep   - Is repeatedly normalized or not (default=False)
```
python fast5_plotter.py -f5 <fast5_dir> -mad 5
```
#### 5. fastq_parser.py
Parse the fastq files inorder shorten the read length or concat together
- Options
  * fq - Path to fastq file
  * shrt - Is fastq shortner (default=True)
  * cat   - Is concat(default=False)
```
python fastq_parser.py -fq <fastq_file> -shrt True
```
#### 6. line_diff.py
Compare 2 files line by line
- Options
  * f1 - The original file
  * f2 - The other file to compare
```
python line_diff.py -f1 <file_1.txt> -f2 <file_2.txt>
```
#### 7. mycopy.py
Custom copy script with enhanced uses to copy/move/delete/rename
- Options
  * s - The path to source of files
  * d - The path to destination to be copied
  * i - Iterate through files including child directories (default=True)
  * ext - File extension type to be copied (default='' -> any)
  * c - Amount of files to be copied/moved/deleted (default=0 -> means all files)
  * cut - Is cut and paste files (default=False)
  * del - Is delete source files (default=False)
  * rn - Suffix to rename files while copying files (default='')
```
python mycopy.py -s <src_dir> -d <dest_dir> -ext fast5 -c 10000 -cut True -rn _zymo
```
#### 8. npy_plotter.py
Plot the numpy dump files
- Options
  * np - The path to npy files directory
  * mad - Median absolute deviation value for data normalization (default=3)
  * rep   - Is repeatedly normalized or not (default=False)
```
python npy_plotter.py -np <npy_dir> -mad 5
```
#### 9. read_fast5.py
Print reads in a fast5 file
- Options
  * ft - The path to fast5 files directory
```
python read_fast5.py -f5 <fast5_dir>
```
#### 10. read_npy.py
Print data array in numpy dump
- Options
  * np - The path to npy file
```
python read_npy.py -np <npy_file>
```
