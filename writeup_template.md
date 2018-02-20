# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeupimages/datavis.PNG "Visualization"
[image2]: ./writeupimages/grayscale.PNG "grayscale"
[image3]: ./writeupimages/normalised.PNG "normalised"
[image4]: ./nd/1.jpg "image1online"
[image5]: ./nd/2.jpg "image2online"
[image6]: ./nd/3.jpg "image3online"
[image7]: ./nd/4.jpg "image4online"
[image8]: ./nd/5.jpg "image5online"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the no. of images available for each class in the dataset.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. The following points describe how the data was preprocessed :

As a first step, I decided to convert the images to grayscale because in this task where we have to classify traffic sign images, the color channels of the image do not 
represent data that is curcial to maintain. Morover it will only make the model learn uneccessary patterns which will cause uneccessary uncertainity while classifying. Theirfore, we are conveting the image to
grayscale.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data becuase genearlly the feautres of images are distributed across all values (0-255), The problem is the while backpropogating
we multiply learning rate with these feature values which will then cause it over or under compensate if they are not normalised to particular range. Hence we 
normalise the image set to particular values using :

img = img/127.5-1

This will ensure smooth learning of model and prevent it from oscillating.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

In order to prevent overfitting between training and vaildation dataset we might need to augment the dataset. It was seen that their were certain classes in the dataset that had ver few
images which reults in model being bias towards other classes. We can augment images using different techniques like rotating, flipping, sharpening etc. However, it was found that 
using dropout the difference between training and validation accuracy was very less thus augementing was left to be implementated some point later. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Dropout Layer			| keep_prob  0.5 for training and 1 for test and validation|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Dropout Layer			| keep_prob  0.5 for training and 1 for test and validation|
| Flattening			| Output -> 400									|
| Fully Connected 1		| 400 input, outputs 120    					| 
| RELU					|												|
| Dropout Layer			| keep_prob  0.5 for training and 1 for test and validation|
| Fully Connected 2		| 120 input, outputs 84					    	|
| RELU					|												|
| Dropout Layer			| keep_prob  0.5 for training and 1 for test and validation|
| Final Output			| 84 inputs, 43 outputs        									|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimiser. I trained the model for 80 epochs with a learning rate of '0.00098' the value of these hyperparameters were decided after a lot of 
hit and trials and finally these values gave a desirable model which met the required specifications. The model also had dropouts after each layer with keep prob o 0.5
for training and 1 for validation and test dataset.

#### 4. The approach takent to implementat model :

A well known architecture known as LE-NET introduced by Yan-leCun was chosen:

* The architure gave good results on while performing on the Mnist data so this architecture was used as base model.
* Firstly rgb images were fed in the model and it was observed that the model was not performing well.
* After converting the images to grayscale and normalising the data the accuracy improved however was still not upto the mark because the model was biased towards particular image classes,for avoiding this dropout were implemented(after each layer) which finally gave a good accuracy that met the required specifications.
* The final model results were:
	Training and validation set accuracy of -> 99.1 %
	validation set accuracy of -> 95.6 %
	Test images accuracy ->  94.6%
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

-The first image might be difficult to classify because it is skewed at an angle
-The second image might be difficult to classify because their are lot of other noise entities available in image.
-The third image might be difficult to classify because their is watermark embedded as noise in the image.
-The fourth image might be difficult to classify because it has another signboard which will create noise.
-The fifth image might be difficult to classify because it is skewed at a certain angle

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Speed limit (60km/h)   						| 
| Road work 			| Road work 									|
| Children crossing		| Children crossing								|
| General caution	    | General caution					 			|
| Priority road			| Priority road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94.6%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 40th cell of the Ipython notebook.

For the first image, the model does not classify correctly stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .34         			| Speed limit (60km/h)   									| 
| .33     				| ahead only 										|
| .11					| Road Work											|
| .04	      			| Stop sign					 				|
| .037				    | Go straight or right      							|


For the second image, the model does classify correctly Road work sign(prob = 0.95). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| Road Work   									| 
| .015     				| Bumpy road 										|
| .008					| General caution											|
| .006	      			| Wild animals crossing					 				|
| .005				    | Bicycles crossing      							|

For the third image, the model does classify correctly children crossing sign(prob = 0.983). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .983         			| Children Crossing   									| 
| .006     				| Bicycles crossing 										|
| .006					| Right-of-way at the next intersection											|
| .001	      			| Beware of ice/snow					 				|
| .0005				    | Road narrows on the right      							|

For the fourth image, the model does classify correctly general caution sign(prob = 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| general caution   									| 
| .0007     			| Traffic signals 										|
| .000018				| Road work											|
| .000017	      		| Pedestrians					 				|
| .00000003				| Right-of-way at the next intersection      							|

For the fifth image, the model does classify correctly Priority road sign(prob = 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Priority Road   									| 
| .0003     			| Roundabout mandatory 										|
| .00000017				| Speed limit (100km/h)											|
| .00000005	      		| Beware of ice/snow					 				|
| .00000004				| Right-of-way at the next intersection      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


