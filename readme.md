# Convolutional Neural Net to predict future ozone levels
This project involved developing a ConvNet to predict future ozone levels
Data obtained from MLS/Aura satellite via GESDISC website 

## Files with relevant code:
- load_data.py - datagenerator for masked model and method to load files
- interpolated_datagenerator.py - data generator for interpolated model
- InterpolatedDenseModel1.ipynb - ConvNet that uses a lienar interpolation 
- DenseModel2.ipynb - ConvNet that uses a mask 
- CustomMSE.py - custom loss functions with which to train the masked model and evaluate the models



### bad coding / hacks to check if running:
- data generators will just concatenate missing channels if they do not meet a hard coded no. channels i.e. line 210-211 in interpolated_datagenerator.py 
- with this i just check how many channels there were in a typical batch without this hard coded concatenation, and then just set the hard coded no. channels (expected_channels) to be that number
- it can be done better but it works 

