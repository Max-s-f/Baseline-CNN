# Convolutional Neural Net to predict future ozone levels
This project involved developing a ConvNet to predict future ozone levels
Data obtained from MLS/Aura satellite via GESDISC website 

## Files with relevant code:
- load_data.py - datagenerator for masked model and method to load files
- interpolated_datagenerator.py - data generator for interpolated model
- InterpolatedDenseModel1.ipynb - ConvNet that uses a lienar interpolation 
- DenseModel2.ipynb - ConvNet that uses a mask 



#### other stuff:

wget --load-cookies C:\Users\simma362\.urs_cookies --save-cookies C:\Users\simma362\.urs_cookies --auth-no-challenge=on --keep-session-cookies --user=moodle --password=Max8450335#! -i C:\Users\simma362\Desktop\Code\KessenichScripts\Link-Lists\HNO3url.txt

Available date range:
2004 d246 - 2024 d027

Enter desired date range as:
[yyyyddd, yyyyddd]


Curr things to change later:
- Interpolated datagenerator: __get_item__ method hard coded so that it will only check for 308 channels

