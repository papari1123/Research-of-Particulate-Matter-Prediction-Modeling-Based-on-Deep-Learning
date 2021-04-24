# Research-of-Particulate-Matter-Prediction-Modeling-Based-on-Deep-Learning

## 1.Summary
1. predicted particulate matter(PM) of next N hour(N=1,4,12,24) in Korea.<br/>
2. used spatiotemporal prediction method considering external pull factor such as wind and china PM.<br/>
3. used CNN, convolutonal-GRU and locally connected layer.<br/>

## 2.Skill
#### Language 
Python
#### OS 
Window, Linux
#### IDE  
Spyder 
#### Framework & Library
Pandas, Nummpy, Keras, Tensorflow
<br/>
## 3.Work flow
<p align="center">
<img src="image/System Flow.png" width=500 height=550 align="center">
</p> 

## 4.Model
<p align="center">
<img src="image/모델.png" width= 500, height = 720></p>

## 5.Code
### Pre-processing

### Training model

## 6.Result
This is an prediction example of one area in Korea divided 8x10 grid. <br/>
<p align="center">
next 1hour<br/>
<img src="image/R1.png"><br/>
next 4hour<br/>
<img src="image/R4.png"><br/>
next 12hour<br/>
<img src="image/R12.png"><br/>
next 24hour<br/>
<img src="image/R24.png"><br/>
</p>

## 7.Recommanded paper to follow this research
1. Li, X., Peng, L., Yao, X., Cui, S., Hu, Y., You, C., & Chi, T. (2017). Long short-term memory neural network for air pollutant concentration predictions: Method development and evaluation. Environmental Pollution, 231, 997–1004.
2. Huang, C. J., & Kuo, P. H. (2018). A deep cnn-lstm model for particulate matter (Pm2.5) forecasting in smart cities. Sensors (Switzerland), 18(7).
3. Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W., & Woo, W. (2015). Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting, In arXiv preprint arXiv:1506.04214
