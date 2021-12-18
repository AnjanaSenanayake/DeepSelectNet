# SquiggleNet-Plus
SquiggleNet Plus is an improved version of the original (https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02511-y) 1D ResNet based model to classify Oxford Nanopore raw electrical signals as target or non-target for Read-Until sequence enrichment or depletion. SquiggleNet Plus provides enhanced model performances.

### Abstract

> Oxford Nanopore Technologies (ONT) introduced MinION, the first portable commercial sequencer of its kind in 2014. The MinION, being a small handheld device, was able to revolutionize the genomics community by means of its superiority in portability. MinION enables access to real-time current signals when a DNA traverses through the pore. ONTs capability to produce long reads is an added advantage in sequencing to have better genome assemblies. But having to sequence unwanted regions of a genome or unwanted species in a given genome pool is one of the main drawbacks of long reads since it would cost a lot of pore time, reducing the efficiency of sequencing. For example, a DNA strand of one million bases (1Mb) in length on average would take about one hour of sequencing. Therefore, improving the speed of targeted sequencing/selective sequencing in ONT platforms is a widely discussed topic. An efficient and real-time analysis of the current signals from the nanopore would enable performing selective sequencing on a DNA sample with the help of ReadUntil API from ONT which can reverse the current flow in a nanopore to eject a sequencing DNA strand and start sequencing for a new strand. However, existing methods on the current signal to targeted reference genome comparisons are timewise costly and hence would give away the real-time capability of sequencing. Therefore, potential improvements to the performance of selective sequencing in many areas are yet to be discovered. Our approach of deep neural network-based selective sequencing would study possibilities to explore the improved methods of selective sequencing with better accuracy and speed compared to state-of-the art methodologies. Recently in basecalling, there have been breakthrough improvements made with deep neural networks over then existing deterministic approaches. Similarly in selective sequencing, use of customized deep neural network architecture would potentially be beneficial in improving accuracy, speed and also as a sample independent methodology. 
## Scripts

#### 1. arguments.py
>The input arguments define the algorithim are collected

#### 2. main.py
>The algorithm is run 

#### 3. mpnn.py
> Preporcessing of the input data (spatial, temporal modelling happens here)

#### 4. rlcore
> The base algorithm implementation. (PPO)

#### 5. learner.py, rlagent.py
> the multi agents are set up  and the preprocessed inputs are fed into the base RL algorithm. 

#### 6. mape
> consists of the RL enviornment implementation 

## Installation
> Create a virtualenv in Python3.5

#### On Linux:
* Open the terminal in the directory where you want to setup the virtual environment.
* Use the following command to create a python3 virtual environment named "env"

```
python3 -m venv env
```

* Use the following command to activate the virtual environment created.

```
source env/bin/activate
```

* Use the following command to leave the virtual environment.

```
deactivate
```

> Installing the list of required dependencies within the created virtual environment.

* Execute the following command to install the list of required dependencies from ```requirements.txt``` file

```
pip3 install -r requirements.txt
```

## Example

* Running the ```main.py``` script with the following arguments will train 3 agents in a ```simple_line``` environment and dump the results into a new directory named 0 in within marlsave directory.

```
python3 main.py --env-name simple_line --num-agents 3 --save-dir 0
```

* Please refer ```arguments.py``` for a complete list of arguments.
* TIP: Redirecting the both stdout and stderr into two separate files would make it easier to debug and to track outputs. In the following example, the standard output stream is redirected to log.txt file while, standard error stream is redirected to log.err.txt file. You could casually ```cat``` the log.txt to see current status.

```
python3 main.py --env-name simple_line --num-agents 3 --save-dir 0 > log.txt 2> log.err.txt 
``` 

To get cleaner version of the output from the log.txt, you could use following command. This will produce the new file ```filtered.txt```

```
cat log.txt | grep "Num success" > filtered.txt
```
