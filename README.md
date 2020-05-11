# covid_from_ctscan
Model which can predict COVID-19 positive case from axial lung CT-scan images.The model may not detect COVID-19 affected patients who are yet to develop pneumonia symptoms.It can only predict COVID-19 pneumonia cases from normal healthy images. 


# Hypothesis
I have noticed in the internet how researchers have noticed white patches in the boundary of lung CT scans.This gave me motivation to develop a model which can detect the probablity of a CT scan,affected with COVID-19.The videos([Video-1](https://youtu.be/3ttAFm9wKPg) and [Video-2](https://youtu.be/xUuNr_EFlBM)) attached explains the hypothesis in a better way.

# Data
For the data collection I have used three public data sources:-
1.[Kaggle](https://www.kaggle.com/luisblanche/covidct)
2.[Github](https://github.com/ieee8023/covid-chestxray-dataset)
3.[Github](https://github.com/shervinmin/DeepCovid/tree/master/data)

I belive that the data is accurate and ensured that  no image is repeated by me while modelling.But still the data collected must represent the data from a larger group of people,which may be available later.


Link to the custom dataset generated is found [here](https://drive.google.com/open?id=1oz2m4DQ4UsKggPm76KKFTqH8Lt8JcxuF).


![sample image](https://github.com/themendu/covid_from_ctscan/blob/master/image_references/screenshots/2020.02.17.20024018-p17-61_3.png)


The above sample image has plentiful noise and it varies from one image to other.The images may have taken at different axial cross sections of different people or repeated images taken at different cross sections of the same person.

# Approach
Modelling is done in PyTorch.Batch size is altered as a large batch size may reduce generalization ability of the model.VGG-19(smaller architecture) didn't come out to be better.Directly modelling with all the given images(raw data) the best results obtained are given below(different techniques to remove under or overfitting were used,resnet-50 architecture was better performing)-

![initial model](https://github.com/themendu/covid_from_ctscan/blob/master/image_references/screenshots/Screenshot%20(38).png)


Now let us play with images,blurring the images and then masking them would rather reduce the noise in the images.


![image transformation](https://github.com/themendu/covid_from_ctscan/blob/master/image_references/screenshots/screenshot.png)


But to do this transformation,I had to blur the images a little,so that the main white clusters that are needed remain in the image.So now the model has more meaningful information to learn(the portion I like it to capture).So different transformations are applied on both classes.In the testing of the model,both these transformations would be applied(done in app.py,similar to TTA) and more certain results are chosen.This need to be done during my validation as labels are known beforehand.

Now a best model on all images was found.Another model trained was initializing the model on all images and trained on the 232 refined images.


![halved](https://github.com/themendu/covid_from_ctscan/blob/master/image_references/screenshots/res_pos_original_2020.01.24.919183-p27-133._a.png_6914a6da-db59-4386-a0fb-a19ff96dc0d1.png)


Now each image was cut into two halves vertically(this was done so that my model can understand more texture based information than spatial information) and then augmented to 1000 images,500 in each class.Resnet-18 model(trained from scratch with no pretrained initializations)on these 1000 images was the best performing one.Now visualizing the layers of the above both models was done.

Upon Visualization the best model is the model trained on halved images as expected though it had lesser accuracy.

# Insights generated
The best model obtained with transfer learning is shown below.It has an accuracy score of 95%(190 correct out of 200 in validation set).

![best_model](https://github.com/themendu/covid_from_ctscan/blob/master/image_references/screenshots/Presentation12.jpg)

This model shown above could detect ALL COVID positive images in validation image set and most importantly it actually learns and generalizes COVID-19 "white spots".The only drawback is that it misdetects non-covid as COVID-19.

Adding weight_decay to an SGD optimizer just adds an L2-regularizer like term during your optimizer step.(So that was added to reduce the overfit).Dropout layerwas introduced in the final fully connected layer as another way to tacckle overfit. 

![Wrong fit](https://github.com/themendu/covid_from_ctscan/blob/master/image_references/screenshots/Presentation1.jpg)

Not only accuracy and recall score,making sure our model does learn the excpected thing is ncessary.The above image explains the beforesaid line,the model overfits to the noise(Gradients show up near the black portion,which is unnecessary) though it's validation scores are better.

So this way my model performs better on the testing dataset as it learnt the thing exactly I wanted to do.Once more data is available,testing the above model is necessary so that it can correctly detect images of various types(age based or may be pneumonia affected lungs etc).

But one thing for sure,this is no replacement to conventional blood testing.Review of my work is most welcome.

# Files Uploaded
I have made a local host website solely for the COVID-19 prediction and the files are uploaded(with app.py as the major file,but the model file cannot be uploaded due to storage issues).

The link to the custom dataset generated in the process is already mentioned above(can be accessed [here](https://drive.google.com/open?id=1oz2m4DQ4UsKggPm76KKFTqH8Lt8JcxuF)).

The three commented code files(two for momdelling and one for layer visualization) are also uploaded.


