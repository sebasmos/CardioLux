# Cardiolux
1. Training set is - CPSC2018 training set, 6,877 recordings: link; MD5-hash: 7b6b1f1ab1b4c59169c639d379575a87 and can be downloaded here: https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_CPSC.tar.gz
2. Evaluation files: https://github.com/physionetchallenges/evaluation-2020

# Run as:
Please to run our code please change the second model folder by CNN_1.model, since keras library only allowed us to save it in this addresss.Thank you: 

* `python train_model.py "../training_data" "./model" `
* `python driver.py "CNN_1.model/" "../test_data" "../results" `

For Cross validation use crossval.py

## Contents

## From Windows/Linux, Compile as - inside the main code folder: 
* `python train_model.py "../training_data" "./model" `
* `python driver.py "./model" "../test_data" "../results" `
## Afterwards, compile scoring - inside the evaluation code folder: 
* `python evaluate_12ECG_score.py "../../test_data" "../../results" scores.csv `

This code uses two main scripts to train the model and classify the data:

* `train_model.py` Train your model. Add your model code to the `train_12ECG_model` function. It also performs all file input and output. **Do not** edit this script or we will be unable to evaluate your submission.
* `driver.py` is the classifier which calls the output from your `train_model` script. It also performs all file input and output. **Do not** edit this script or we will be unable to evaluate your submission.

Check the code in these files for the input and output formats for the `train_model` and `driver` scripts.

To create and save your model, you should edit `train_12ECG_classifier.py` script. Note that you should not change the input arguments of the `train_12ECG_classifier` function or add output arguments. The needed models and parameters should be saved in a separated file. In the sample code, an additional script, `get_12ECG_features.py`, is used to extract hand-crafted features. 

To run your classifier, you should edit the `run_12ECG_classifier.py` script, which takes a single recording as input and outputs the predicted classes and probabilities. Please, keep the formats of both outputs as they are shown in the example. You should not change the inputs and outputs of the `run_12ECG_classifier` function.

## Use

You can run this classifier code by installing the requirements and running

    python train_model.py training_data model   
    python driver.py model test_data test_outputs

where `training_data` is a directory of training data files, `model` is a directory of files for the model, `test_data` is the directory of test data files, and `test_outputs` is a directory of classifier outputs.  The [PhysioNet/CinC 2020 webpage](https://physionetchallenges.github.io/2020/) provides a training database with data files and a description of the contents and structure of these files.
## Branch NN_1
Keras models are saved without imputer and classes variables since they are added within `run_12ECG_classifier.py`
To run:
* `python train_model.py "../training_data" "./model" `
* `python driver.py "NN_1.model/" "../test_data" "../results" `
Inside of ./model we will save the imputer, which will contain relevant information regarding the features from the trained model and that shall be used inside the testing set
## Branch windowing
This branch contains the `fun_extract_data` which contains a windowing functionality with non-linear features
## Submission

The `driver.py`, `get_12ECG_score.py`, and `get_12ECG_features.py` scripts must be in the root path of your repository. If they are inside a folder, then the submission will be unsuccessful.

## Details

See the [PhysioNet/CinC 2020 webpage](https://physionetchallenges.github.io/2020/) for more details, including instructions for the other files in this repository.

