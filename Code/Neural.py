import numpy as np
import pandas as pd
from sklearn import preprocessing

raw_csv=np.loadtxt('neural.csv',delimiter=',')
unscaled=raw_csv[:,:-1]
targets=raw_csv[:,-1]

num_one=0
indices_to_remove=[]
for i in range(targets.shape[0]):
    if targets[i]==1:
        num_one+=1
        if num_one>305:
           indices_to_remove.append(i)

unscaled_equal_priors=np.delete(unscaled,indices_to_remove,axis=0)
targets=np.delete(targets,indices_to_remove,axis=0)

scaled_inputs=preprocessing.scale(unscaled_equal_priors)
#scaled_inputs=unscaled
#shuffle

shuffled_indices=np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs=scaled_inputs[shuffled_indices]
shuffled_targets=targets[shuffled_indices]

samples_count=shuffled_inputs.shape[0]
train_samples_count=int(0.8*samples_count)
validation_samples_count=int(0.1*samples_count)
test_samples_count=samples_count-train_samples_count-validation_samples_count

train_inputs=shuffled_inputs[:train_samples_count]
train_targets=targets[:train_samples_count]

validation_inputs=shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets=targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs=shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets=targets[train_samples_count+validation_samples_count:]

w=pd.DataFrame(test_inputs)
w.to_csv('check.csv')

np.savez('Loan_data_train',inputs=train_inputs,targets=train_targets)
np.savez('Loan_data_validation',inputs=validation_inputs,targets=validation_targets)
np.savez('Loan_data_test',inputs=test_inputs,targets=test_targets)
