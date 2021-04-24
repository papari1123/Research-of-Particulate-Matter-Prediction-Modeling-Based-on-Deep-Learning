# Research-of-Particulate-Matter-Prediction-Modeling-Based-on-Deep-Learning

## 1.Summary
 This study proposes the PM prediction modeling based on **deep learning by using ConvGRU which can simultaneously analyze spatiotemporal information, and using a locally-connected layer which can better extract features of individual fields.** 

 Experiments were designed **to predict the PM10 of next 1, 4, 12 and 24 hours with the spatial resolution divided by the 8x10 grid of all regions in Korea.** In order to verify the performance of the proposed model, this study made five experimental hypotheses, which confirm that the proposed model is better than the other deep-learning based prediction model. 

 In the result, the prediction performance got better when
 1) It analyzed **spatiotemporal information simultaneously.** 
 2) It had low computational complexity for short-term prediction; and it has high complexity for long-term prediction.
 3) It considered a intermediate process up to the next 1 hour to predict the next T-1 hour.
 4) It considered factors of PM diffusion in Korea.
 5) It considered factors of China PM. 
  
 So, the proposed model showed the better prediction performance than the previously studied models. Also, the result showed the delay shift phenomenon in the short-term prediction, and showed the moving average in the long-term prediction. So, the study can conclude that the prediction performance can be improved if those phenomenon are solved.

## 2.Skill
#### Language 
Python
#### OS 
Window, Linux
#### IDE  
Spyder 
#### Framework & Library
Pandas, Numpy, Keras, Tensorflow
<br/>

## 3.Work flow
<p align="center">
<img src="image/System Flow.png" width=500 height=550 align="center">
</p> 

## 4.Hyper parameter
|Parameter|Value|
|:---:|:---:|
|Training data|60% (2014~2016)|
|Validation data|20% (2017)|
|Testing data|20% (2018)|
|Prediction length (T, hour)|[1, 4, 12, 24]|
|History length (hour)|24|
|Time interval (hour)|1|
|Optimizer|Amsgrad|
|Learning rate|0.00075|
|Max training epochs|100|
|Loss function|Mean square error|
|Callback method|Early stopping with patience = 10|

## 5.Used data
#### raw data
1. pollution data in Korea : https://www.airkorea.or.kr/web/last_amb_hour_data?pMENU_NO=123.
2. meteorological data in Korea : https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36.
3. particulate matter in China : http://www.stateair.net/web/post/1/1.html

#### model input data
|#|Variable|Dimension (C input)|T input|W input|C input|
|:---:|:---:|:---:|:---:|:---:|:---:|
|0|PM10(μg/m3)|scalar(7)|O|O|O|
| 1  | SO2(ppm)                    | scalar         | O        | -        | O        |
| 2  | NO2(ppm)                    | scalar         | O        | -        | O        |
| 3  | Temperature(℃)                    | scalar         | O        | -        | O        |
| 4  | Wind speed(m/s)                     | scalar         | O        | O        | O        |
| 5  | Humidity(%)                 | scalar         | O        | O        | O        |
| 6  | Air pressure(hPs)                     | scalar         | O        | -        | O        |
| 7  | Wind_u factor(m/s)                  | scalar (7)     | -        | O        | O        |
| 8  | Wind_v factor(m/s)                 | scalar (7)     | -        | O        | O        |
| 9  | Precipitation(cm)                    | scalar         | O        | -        | O        |
| 10 | Daily largest temperature difference(℃)                     | scalar         | O        | -        | O        |
| 11 | Daily minimum temperatrue(℃)                  | scalar         | O        | -        | O        |
| 12 | Beijing PM2.5(μg/m3)             | 24             | -        | -        | O        |
| 13 | Season                      | 4              | O        | -        | O        |
| 14 | hour                    | 6              | O        | -        | O        |
| 15 | X axis in grid                    | scalar         | O        | -        | O        |
| 16 | Y axis in grid                    | scalar         | O        | -        | O        |


## 6.Model
<p align="center">
<img src="image/모델.png" width= 450, height = 720></p>

## 7.Code
### Pre-processing
(TBD)
### Training model
(TBD)
## 8.Result
This is an prediction example of one area in Korea divided 8x10 grid. <br/>
- 1st subplot represents a result of one year(2018).<br/>
- 2nd subplot represents a result of first month(2018.1).<br/>
- 3rd subplot represents a result of one month in middle of year.<br/>
- last subplot represents a result of last month(2018.12).<br/>
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

## 9.Recommanded paper to follow this research
1. Li, X., Peng, L., Yao, X., Cui, S., Hu, Y., You, C., & Chi, T. (2017). Long short-term memory neural network for air pollutant concentration predictions: Method development and evaluation. Environmental Pollution, 231, 997–1004.
2. Huang, C. J., & Kuo, P. H. (2018). A deep cnn-lstm model for particulate matter (Pm2.5) forecasting in smart cities. Sensors (Switzerland), 18(7).
3. Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W., & Woo, W. (2015). Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting, In arXiv preprint arXiv:1506.04214
