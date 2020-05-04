# covid_from_ctscan
Model which can predict COVID-19 positive case from axial lung CT-scan images.This work is my personal project and I in no way claim my work can be used universally,or state that my work is accurate.


# Hypothesis
I have noticed in the internet how researchers have noticed white patches in the boundary of lung CT scans.This gave me motivation to develop a model which can detect the probablity of a CT scan,affected with COVID-19.The videos([a](https://youtu.be/3ttAFm9wKPg) and [b](https://youtu.be/xUuNr_EFlBM)) attached explains in a better way.

# Data
For the data collection I have used three public data sources:-
1.[Kaggle](https://www.kaggle.com/luisblanche/covidct)
2.[Github](https://github.com/ieee8023/covid-chestxray-dataset)
3.[Github](https://github.com/shervinmin/DeepCovid/tree/master/data)

I belive that the data is accurate and ensured that  no image is repeated by me while modelling.


My custom dataset is found [here](https://drive.google.com/open?id=1oz2m4DQ4UsKggPm76KKFTqH8Lt8JcxuF).
![sample image](https://drive.google.com/file/d/1yXu4dZ17ZnYSGZbEoBes9yBYv7Z98-Mg/view?usp=sharing)
The above sample image has plentiful noise and it varies from one image to other.The images may have taken at different axial cross sections of different people or repeated images taken at different cross sections of the same person.

# Approach
While playing with images,blurring the images and then masking them would rather reduce the noise in the images.
![image transformation](https://drive.google.com/open?id=1DbzoDS8kIsqyv2YivNsaLeqfWHRFcxj3)
But to do this transformation,I had to blur the images a little,so that the main white clusters that are needed remain in the image.So now the model has more meaningful information to learn(the portion I like it to capture).So I applied different transformations on both classes.In the testing of the model,both these transformations would be applied(done in app.py,similar to TTA) and more certain results are chosen.This need to be done during my validation as labels are known beforehand.
![results with larger data](https://drive.google.com/file/d/1wOedGapWChI1eIoy604-f60b9jbtKDYH/view?usp=sharing)
Modelling is done in PyTorch.Batch size is altered as a large batch size may reduce generalization ability of the model.VGG-19(smaller architecture) didn't come out to be better.Resnet-50 was used to finalize the reuslts.Recall score was monitored at every epoch.Training on a large dataset is used and the best performing model was saved.Now a plethora of options are available.So these previous best parameters were intialized on the resnet-50 model trained  only on perfectly filtered images(232 out of 582) while training only the last two layers and fc layer of resnet-50(people refer it as transfer learning).Resnet-50 has 4 layers(each layer made up of several bottleneck layers-with batch norm included).This fetched me best accuracy.

# Insights generated
The best model obtained with transfer learning is shown below.It can  successfully detect  20 for every 21 COVID-19 scans given as input.![best_model](https://drive.google.com/file/d/1et33a_gDCSS3RJd0NPa3Wtz0908yR-OP/view?usp=sharing).This model could detect 20 out of 21 COVID positive images in validation images along with correcly predicting 19 out of 19 COVID-negative images.

Adding weight_decay to an SGD optimizer just adds an L2-regularizer like term during your optimizer step.(So that was added to reduce the overfit.) 

Not only accuracy,making sure our model does learn the excpected thing is ncessary.

Once more data is available,testing the above model is necessary so that it can correctly detect images of various types(age based or may be pneumonia affected lungs etc).I also need to make sure my model did not overfit on the training set.

