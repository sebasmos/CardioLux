import os
import shutil
import random
import fnmatch

from train_12ECG_classifier import train_12ECG_classifier
# data folder contains dataset (hea + mat)

k = 5

shutil.copytree('../dataset', 'temp')
headers = fnmatch.filter(os.listdir('temp'), "*.hea")
N = len(headers)

for i in range(0, k): # Creates folds
    os.mkdir('fold_%s' % (i))
    headers = fnmatch.filter(os.listdir('temp'), "*.hea")
    foldsize = round(N/k)

    if i == k - 1: foldsize = len(headers)

    files = random.sample(headers, foldsize) # @TODO: folding shoul consider HEA+MAT couples
    for file in files:
        shutil.move(f"temp/{file}", 'fold_%s' % (i))
        shutil.move(f"temp/{file.replace('.hea','.mat')}", 'fold_%s' % (i))

os.rmdir('temp')

os.mkdir('output')

for i in range(0, k):    
    os.mkdir('training') # Temporal training folder
    shutil.copytree('fold_%s' % (i), 'test') # Copies i fold to test folder
    
    for j in range(0, k):
        if j != i:
            files = os.listdir('fold_%s' % (j))
            for file in files:
                shutil.copy(f"fold_{j}/{file}", 'training')
     
    os.system('python train_model.py "training" "./model"') 
    os.system('python driver.py "CNN_1.model" "test" "output"') 
     #python driver.py "CNN_1.model" "test" "../results" `
     #train_model "training" "./model"
     #train_12ECG_classifier("training", "./model")# Call training.py with training folder to train the model. This should update the model
     #driver "CNN_1.model" "test" "../output"
     #driver('test', 'output') # Call driver with test and output folder
    # Run evaluation code on output folder
    os.system('python evaluate_12ECG_score.py "test" "output" scores_%s.csv' % (i))   
    shutil.rmtree('test')
    shutil.rmtree('training')

 


# Cleaning
for i in range(0, k):
    shutil.rmtree('fold_%s' % (i))
#os.rmdir('output')
