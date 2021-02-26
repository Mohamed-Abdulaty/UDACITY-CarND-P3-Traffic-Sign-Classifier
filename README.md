# **Traffic Sign Recognition**  [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### *Solution of Project #3 of Udacity's Self Driving Car Nanodegree.*
#


### **Goals & steps of the project**
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pics/histogram.png                         "Visualization"
[image2]: ./pics/pre_preprocessing.png                 "original"
[image3]: ./pics/post_preprocessing.png                "preprocessed"
[image4]: ./pics/new_images.png                        "new Traffic Signs" 
[image5]: ./pics/prop_speed_limit_30.png               "prop_speed_limit_30"
[image6]: ./pics/prop_priority_road.png                "prop_priority_road"
[image7]: ./pics/prop_yield.png                        "prop_yield"
[image8]: ./pics/prop_stop.png                         "prop_stop"
[image9]: ./pics/prop_no_entry.png                     "prop_no_entry"
[image10]: ./pics/prop_slippery_road.png               "prop_slippery_road"
[image11]: ./pics/prop_wild_animals_crossing.png       "prop_wild_animals_crossing"
[image12]: ./pics/prop_ahead_only.png                  "prop_ahead_only"


---

### **Data Set Summary & Exploration**

#### **1. Summary of the data set**

* *I used the pandas library to calculate summary statistics of the traffic
signs data set:*  

  * The size of training set is *34799*
  * The size of the validation set is *4410*
  * The size of test set is *12630*
  * The shape of a traffic sign image is *(32, 32, 3)*
  * The number of unique classes/labels in the data set is *43*

---
#### **2. Exploratory visualization of the dataset**

* *The bar chart shows the data distribution of the training data.*
* *Each bar represents one class (traffic sign) - indicated bellow the chart- and how many samples are in the class.*
* *The mapping of traffic sign names to class id can be found in [signnames.csv](./signnames.csv)*

![alt text][image1]

---

### **Design and Test a Model Architecture**

#### **1. Preprocessing**

* Knowing that each image in the training dataset have 3 color channels RGB. I have transform the image firstly to the *'YCrCb'* color space, and afterwards extract only the *'y-channel'* channel. 
* Afterwards, I have normalized the data before training.

|original image     |preprocessed image |
|-------------------|-------------------|
|![alt text][image2]|![alt text][image3]|
|                   |                   |

#### **2. Model Architecture**
 
* Using convolutional neuronal network to classify the traffic signs. 
* The input of the network is an 32x32x1 image and the output is the probability of each of the 43 possible traffic signs.
 
*The model consisted of the following layers:*

| Layer         		| Description	             					| Input     | Output    | 
|:----------------------|:----------------------------------------------|:----------|:----------|
| Convolution 5x5     	| 1x1 stride, valid padding, RELU activation 	| 32x32x 1  | 28x28x48  |
| Max pooling			| 2x2 stride, 2x2 window						| 28x28x48  | 14x14x48  |
| Convolution 5x5 	    | 1x1 stride, valid padding, RELU activation 	| 14x14x48  | 10x10x96  |
| Max pooling			| 2x2 stride, 2x2 window	   					| 10x10x96  |  5x 5x96  |
| Convolution 3x3 		| 1x1 stride, valid padding, RELU activation    |  5x 5x96  |  3x 3x172 |
| Max pooling			| 1x1 stride, 2x2 window        				|  3x 3x172 |  2x 2x172 |
| Flatten				| 3 dimensions -> 1 dimension					|  2x 2x172 | 688       |
| Fully Connected       | connect every neuron from layer above			| 688       | 84        |
| Fully Connected       | output = number of traffic signs in data set	| 84        | 43        |
|                       |                                               |           |           |



#### **3. Model Training**

*I have used the Udacity workspace for training.*

Here are my training parameters:  
  * EPOCHS = 35
  * BATCH_SIZE = 128
  * SIGMA = 0.1
  * OPTIMIZER: AdamOptimizer  
    * learning rate = 0.001


#### **4. Solution Approach**

* First implementation was LeNet-5 shown in the Udacity classroom.  
  * With small modifications it worked with the input shape of 32x32x3.  
  * It was a good starting point and the validation accuracy was about 90%. 
  * However, the test accuracy was much lower (less that 80%).

* Afterwards, I modified the network and added more convolutional layer, did some preprocessing.  
  * I have been playing with both *learning rate* and the number of *Epochs* for sometime.
  * I have tried multiple methods for preprocessing such as:  
    * Gamma correction
    * Gaussian normalization
    * [Unsharpining using the laplacian](https://www.idtools.com.au/unsharp-masking-python-opencv/)
  * Most of the preprocessing methods mentioned above were not effecting the result in a good way. Therefore, I decided to keep it simple.
  * I have used some ideas also for these papers:  
    * [CNN Design for Real-Time Traffic Sign Recognition](https://www.sciencedirect.com/science/article/pii/S1877705817341231)  
    * [Traffic Signs Detection and Tracking using Modified Hough Transform](https://www.scitepress.org/Papers/2015/55432/55432.pdf)
    It was not a success unfortunately.

### **Test a Model on New Images**

#### **1. Getting New Images**

I used google to get new images for my testing set. Here are 10 examples I collected.

![alt text][image4]

The signs are:  
* Speed limit 30
* Priority road
* Yield
* Stop
* No entry
* Slippery road
* Wild animals crossing
* Ahead only

Some of these signs are tricky ... 

Reasons:
* The priority road sign
  * It is not a 100% complete sign
* The wild animal crossing sign  
  * It is not the same standard sign. It has different shape and colors.  
* The slippery road sign
  * It is - also - not the same standard sign. It has different shape and colors.




#### 2. Performance on New Images

| Image			            |     Prediction		    | 
|:---------------------:|:---------------------:| 
| Speed limit 30        | Speed limit 30        | 
| Priority road   		  | Priority road 	      |
| Yield			            | Yield					        |
| Stop		              | Stop					        |
| No entry		          | No entry              |
| Slippery road         | Slippery road         |
| Wild animals crossing | Wild animals crossing |
| Ahead only            | Ahead only            |
|                       |                       |

* *Comparing the testing set results from earlier vs. the new images we concluded the following:*
  * Testing set accuracy = 95.1%  
  * New images accuracy  = 62.5% *(5 of 8 correct)*

#### **3. Softmax Probabilities**

**Predictions**


| Result              | Probability | Prediction	       | Correctness |
|:--------------------|:------------|:---------------------|:------------|
| ![alt text][image5] | 1.00        | Speed limit 30   	   | True        |
| ![alt text][image6] | 0.70     	| Keep left     	   | False       |
| ![alt text][image7] | 1.00		| Yield				   | True        |
| ![alt text][image8] | 1.00	    | Stop				   | True        |
| ![alt text][image9] | 1.00		| No entry      	   | True        |
| ![alt text][image10]| 1.00		| Roundabout mandatory | False       |
| ![alt text][image11]| 1.00		| Roundabout mandatory | False       |
| ![alt text][image12]| 1.00		| Ahead only           | True        |
|                     |             |                      |



