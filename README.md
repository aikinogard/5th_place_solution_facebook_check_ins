# 5th_place_solution_facebook_check_ins

My solution rank 5th/1212 in Facebook check ins prediction competition at Kaggle

## Instruction

* First download data from [Kaggle](https://www.kaggle.com/c/facebook-v-predicting-check-ins/data) into `data` folder
* run `split.py` in `data` folder to split validation set.
* run each model in `single_model`. Just run `python model.py [config_name in models.config]`
    * `valid_mode_on=True` for validation mode, which provide you local validation score on the middle 1/100 data.
    * `valid_mode_on=False` for prediction mode, which run on the whole dataset and does not return local validation score.
* `ensemble` folder provides code for ensemble. Just run `python ensemble.py [config_name in ensembles.config]`.
    * Validation mode and prediction mode are also available like in single model.
    * (optional) `ensemble_add.py` and `ensemble_remove.py` test the ensmeble combination by adding or removing one single model.
    * (optional) `ensemble_bayes.py` use bayes optimization to find the optimal soft-voting weight.

-----------
This is an amazing competition and thank all the discussion and scripts made by fellow kaggler. I joined two weeks before the end of this competition. The main reason for me to start this competition is because:

* The features provided in this competition are very simple (only x, y, time, accuracy). But the data set is big and contains more than 100,000 labels. This gives me chance to think about how to divide the problem into small sub-problems.

* Get into top 10 in order to reach master tier.

I learned a lot during this competition. Especially in the final week, the competition become quite intense among ranking 5th to 10th. The fear of fall out of top 10 making me work very hard on trying and learning new method, including naive bayes, support vector machine and kernel density estimation. Special thank to @Qingchen, it is always very educative to talk with him.

## validation set
The size of the data set is very large. In order to use my time efficiently, my model and hyperparameter selection are done on 1/100 of the total data set. It is data in range 5 < x < 6, 5 < y < 6. Then I use the last 171360 minutes as validation set. By using this subset, I am able to know the performance of one model within 20 minutes.

According to the admin, the real test set has removed unseen place_id. But I didn't do so on my validation set. So my local map3 will be lower than public leaderboard.

## strategy of hyperparameter selections
n_cell_x, int
    number of cells divided in x for test and validation set
n_cell_y, int
    number of cells divided in y for test and validation set
x_border, float
    border on the edge of the cell for larger range of training set
y_border, float
    border on the edge of the cell for larger range of training set
th, int
    cutoff low frequency labels
Those parameters as well as parameters of different models are considered as the hyperparameters. I write my own random grid search code to find the optimal hyperparameters of each classifier. This code will append all the hypermeters and corresponding local validation score to a record file. It also guarantee new generated hyperparameters will not duplicate with previous one in record. Thus, I can easily adjust the range and density of each hyperparameters in the selection process.

## ensemble
Each single model will predict top 10 place_id with probabilities for each row_id. The are stored in csv as (row_id, place_id, proba). This output file for a validation set is about 3.3Mb and 351Mb for test set. I ensemble probabilities of different models using python defaultdict and output the top 3 place_id as prediction. I will first ensemble the validation set to see which combination of models has better performance. Then, I will run an ensemble on the whole data set with this combination. The idea of finding the combination is:

* Try to increase the diversity of the model. This includes different number of cells and different models.

* Then I list model candidates in different category and sorted them by local map3 score.

* Do greedy search to find the combination with highest local validation score.

* My final submission is an unweighted sum of single models. I also tried to find weight by bayes optimization. There is some improvement on the 5th digit so I just ignore it.

* Then run the ensemble on the whole dataset.

## models in my ensemble
My final submission is a soft-voting of 13 single models, including k-nearest neighbors, random forest, extra-trees, gradient boosting trees, naive bayes, kernel density estimation.

    sklearn.neighbors.KNeighborsClassifier/knn_ps2_n1, local map3=0.504975
    
    sklearn.ensemble.RandomForestClassifier/rf_opt1, local map3=0.509508
    
    sklearn.ensemble.ExtraTreesClassifier/et_opt1, local map3=0.497024
    
    XGBoost/xgb0_n4, local map3=0.518815
    XGBoost/xgb0_n2, local map3=0.514897
    XGBoost/xgb0_n4_th13_border5, local map3=0.521114
    XGBoost/xgb150opt, local map3=0.523222
    XGBoost/xgb150opt2, local map3=0.523853
    
    sklearn.naive_bayes.GaussianNB/nb_xyat_weight1_n9, local map3=0.482681
    
    sklearn.neighbors.KernelDensity/kde_opt1, local map3=0.478387
    scipy.stats.gaussian_kde/kde_opt2c, local map3=0.491391
    scipy.stats.gaussian_kde/kde_opt3c, local map3=0.493110
    KDE-for-SciPy/kde_opt4, local map3=0.483254

## Insights
The structure of this dataset is very interesting. My best XGBoost model only has max_depth=3, which is unusual for my experience in previous competitions. This indicates features are almost independent to each other in this dataset. This can be the reason why my Naive Bayes is only 4% worse than my knn.

Naive Bayes is a first assumption of the data distribution. When using the sklearn.naive_bayes.GaussianNB, it also assumes the distribution of each feature in cell is unimodal. But doing a simple visualization of the x, y for given place_id will tell you that assumption is not very good. One way to improve the Naive Bayes is to use kernel density estimation (KDE) rather than one single gaussian to get the probability density function. I recommend this blog (http://www.mglerner.com/blog/?p=28) for nice explanation of KDE.

The performance of KDE is determined by the kernel and kernel parameters (bandwidth for gaussian kernel). **sklearn.neighbors.KernelDensity** provides you plenty of kernels to choose. My hyperparameter selection code told me gaussian and exponential are optimal choice while tophat, linear, cosine works badly. sklearn does not provide rule of thumb estimation of bandwidth, so the bandwidth is selected by cross-validation. I also tried the **scipy.stats.gaussian_kde** because it provides two common rule of thumb estimations of bandwidth by statistician, "scott" and "silverman". They can give fast estimation but are designed for unimodal distribution. In order to get better bandwidth estimation, I also used Daniel B Smith's implement of KDE (https://github.com/Daniel-B-Smith/KDE-for-SciPy). **KDE-for-SciPy** is based on this paper
    > Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
    > estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

This estimation is more time consuming but can give good estimation on multimodal distribution. My final submission include KDE from all those implements.

-----------------
I feel it is very important to have your own framework to do the model selections and ensemble. The quality of your own code determine how easy it is to tuning the model and test new idea. Thus, this competition is also a good programing exercise for me.
