# covid_from_ctscan
Deep Learning-based solution which can predict pneumonic(may be caused due to COVID-19) white spots from axial lung CT-scan images. This work in any way does NOT claim any perfect method of detection.


# Hypothesis
Researchers have noticed white patches in the boundary of lung CT scans in COVID-affected patients identifying it as pneumonia. This is the motivation to develop a model which can detect the probability of a CT scan affected with COVID-19.The videos([Video-1](https://youtu.be/3ttAFm9wKPg) and [Video-2](https://youtu.be/xUuNr_EFlBM)) attached explains the hypothesis in a better way.

# Data
Three public data sources:-
1.[Kaggle](https://www.kaggle.com/luisblanche/covidct)
2.[Github](https://github.com/ieee8023/covid-chestxray-dataset)
3.[Github](https://github.com/shervinmin/DeepCovid/tree/master/data)

Link to the custom dataset generated is found [here](https://drive.google.com/open?id=1oz2m4DQ4UsKggPm76KKFTqH8Lt8JcxuF).


![sample image](https://github.com/themendu/covid_from_ctscan/blob/master/image_references/screenshots/2020.02.17.20024018-p17-61_3.png)


The above sample image has high noise (portion outer to the lungs), and it varies from one image to another.

![sampled image](https://github.com/themendu/covid_from_ctscan/blob/master/image_references/screenshots/new.jpg)

The images may have been taken at different axial cross-sections of different people, or repeated images were taken at different cross-sections of the same person. So as per the labeling above, the part between the lungs can be considered as noise.

# Bayesian Neural Network for Out Of Distribution Detection
Resnet-18 architecture is used as a feature detector and the fully connected layer as bayesian architecture with prior normal distributions to every parameter in the FC layer. So by giving the same input again and again to the model, OOD samples can be separated. Every layer is trained. Average KL divergence loss for every layer is also optimized along with cross-entropy to impose a sparsity constraint.


# Approach
Batch size is altered as a large batch size may reduce the generalization ability of the model. Directly training with all the given images(raw data), the best results obtained are given below -

![initial model](https://github.com/themendu/covid_from_ctscan/blob/master/image_references/screenshots/Screenshot%20(38).png)


Blurring the images and then masking them would somewhat reduce the noise in the images.


![image transformation](https://github.com/themendu/covid_from_ctscan/blob/master/image_references/screenshots/screenshot.png)


Now the model has more meaningful information to learn. So different transformations are applied to both classes. In testing the model, both these transformations would be applied (done in app.py, similar to TTA), and more reliable results are chosen. Now the best model on all images was found. Another model trained was initializing the model on all images and trained on the 232 refined images. (let us call it Model-a)


![halved](https://github.com/themendu/covid_from_ctscan/blob/master/image_references/screenshots/res_pos_original_2020.01.24.919183-p27-133._a.png_6914a6da-db59-4386-a0fb-a19ff96dc0d1.png)


Now each image was cut into two halves vertically (this was done so that the model could understand more texture-based information than spatial information) and then augmented to 1000 images,500 in each class.Resnet-18 model(trained from scratch with no pretrained initializations) on these 1000 images was the best performing one. Now visualizing the layers of the above both models was done. (let us call it Model-b).

Upon Visualization, the best model is trained on halved images as expected though it had lesser accuracy (model b).

# Insights generated
The best model (Model-b) is shown below. It has an accuracy score of 95% (190 correctly predicted out of 200 in the validation set).

![best_model](https://github.com/themendu/covid_from_ctscan/blob/master/image_references/screenshots/Presentation12.jpg)

This model shown above could detect all pneumonic COVID positive images in the validation image set, and most importantly it learns and generalizes pneumonic "white spots."

Adding weight_decay to an SGD optimizer adds an L2-regularizer like term during the optimizer step (That was added to reduce the overfit). The dropout layer was introduced in the final fully connected layer as another way to tackle overfitting. The reason why Model-a is not considered best is explained below-

![Wrong fit](https://github.com/themendu/covid_from_ctscan/blob/master/image_references/screenshots/Presentation1.jpg)

Not only accuracy and recall score, making sure our model does learn from the data is necessary. The above image explains the before said line. The model overfits the noise (Gradients show up near the black portion,which is unnecessary) though its validation scores are better.

# Local Host Deployment
A local host is created. More details are in the local-host folder.

# Files Uploaded
A local host website solely for the COVID-19 prediction and its files (with app.py as the major file).

The four commented code files(two for modeling and one for layer visualization, and one for bayesian NN).


