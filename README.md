# CNN Architecture for Pupil Center Estimation in eye images extracted from head tracker of a smartphone
*5 Convolutional Layers with a stride of 1 and 3x3 filters</br>
* 1 Fully Connected Layer with 2048 units</br>
* Average Pooling of 2x2 with a stride of 2</br>
* Batch Normalization and Dropout after every layer</br>
* Loss Function – Euclidean Distance between true and predicted labels</br>

# Raw images of the pupil obtained from a infrared camera in a smartphone along with their histograms
<img src="https://github.com/soumilchugh/PupilCenterEstimation/blob/master/eye1.png" height="300" width="200"> <img src="https://github.com/soumilchugh/PupilCenterEstimation/blob/master/histogram1.png" height="300" width="600"/>
<img src="https://github.com/soumilchugh/PupilCenterEstimation/blob/master/eye1.png" height="300" width="200"> <img src="https://github.com/soumilchugh/PupilCenterEstimation/blob/master/histogram2.png" height="300" width="600">

# Result of applying different image enhancement techniques such as Histogram Equalisation, Power Law, Adaptive Histogram Equalisation
* Adaptive Histogram Equalisation
<img src="https://github.com/soumilchugh/PupilCenterEstimation/blob/master/clahe.png" height="300" width="200">
* Power Law
<img src="https://github.com/soumilchugh/PupilCenterEstimation/blob/master/Powerlaw.png" height="300" width="200">
* Histogram Equalisation
<img src="https://github.com/soumilchugh/PupilCenterEstimation/blob/master/histequal.png" height="300" width="200">
* Adaptive Histogram Equalisation plus Power Law
<img src="https://github.com/soumilchugh/PupilCenterEstimation/blob/master/clahe_powerlaw.png" height="300" width="200">

# Mean Pixel Error after training on the same dataset but with different image enhancement techniques
<img src="https://github.com/soumilchugh/PupilCenterEstimation/blob/master/result.png" height="300" width="500">

