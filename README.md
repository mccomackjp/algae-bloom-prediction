# Algae Bloom Prediction

For my senior design project at MSOE, my team and I used CNNs and  logistic regression models to predict algae blooms in Utah Lake.


## Setup

Download and install [Anaconda]([https://www.anaconda.com/](https://www.anaconda.com/)).


Test that required libraries are installed, by pulling the algae bloom prediction repository and run the Jupyter Notebook entitled 'logistic_regression.ipynb'. To do this run the following steps:

1.  Open up a command line to where the repository was pulled to.
2.  Type Jupyter notebook and press 'enter'
3.  Navigate to the notebook and select the notebook.

1.  The current place will be (<ROOT>/notebooks/models/logistic_regression.ipynb)

5.  From the menu '_Cell_' select, 'Run all'
6.  If the last cell completes, congratulations! You are now ready to Develop and improve the current model!

## Framework Overview

We use a greedy feature selection algorithm and a sliding window process for extracting a single data point from a subset or window within a time series to achieve improved logistic regression model performance.

### Libraries

We use the Python pandas and scikit-learn libraries for data analysis and model development. Scikit-learn offers a common API for various machine learning algorithms to allow for fast and simple algorithm swapping and comparison. We're using logistic regression models, the formula for logistic regression is defined as follows:

![](/images/lr_formula)

where ŷ is the predicted probability, b is the bias or intercept, xi  are the input features, wi  are the feature weights, n is the number of samples, and  ![](https://msoese.atlassian.net/wiki/download/thumbnails/853016632/image2019-5-5_14-31-13.png?version=1&modificationDate=1557084676187&cacheVersion=1&api=v2&width=40&height=27) is the Sigmoid function:

![](https://msoese.atlassian.net/wiki/download/attachments/853016632/image2019-5-5_13-51-7.png?version=1&modificationDate=1557082270064&cacheVersion=1&api=v2)

### Metrics

Accuracy, recall, and precision will be used as the primary metrics for model evaluation:

![](https://msoese.atlassian.net/wiki/download/attachments/853016632/image2019-5-5_13-56-33.png?version=1&modificationDate=1557082596237&cacheVersion=1&api=v2)

![](https://msoese.atlassian.net/wiki/download/attachments/853016632/image2019-5-5_13-56-45.png?version=1&modificationDate=1557082608757&cacheVersion=1&api=v2)

![](https://msoese.atlassian.net/wiki/download/attachments/853016632/image2019-5-5_13-56-59.png?version=1&modificationDate=1557082622523&cacheVersion=1&api=v2)

where tp is the number of true positives (a bloom is predicted a bloom), tn is the number of true negatives (a non-bloom is predicted a non-bloom), fp is the number of false positives (a non-bloom is predicted a bloom), fn is the number of false negatives (a bloom is predicted a non-bloom), ŷi is the predicted class (i.e. bloom or non-bloom), yi  is the actual class, and n is the number of samples.

### Greedy Algorithm

A simple greedy algorithm is used to select features for a given model, the steps the of the algorithm are as follows:

1.  Sort all features based on accuracy (this will be accomplished by training a model with a single feature for each feature in the data set).
2.  Build a null model as the starting base model (Note: a null model has all features set to 0, which is just the intercept in the logistic regression equation. Any set of features can be prescribed as the base model, the default null model is essentially a model with no input features).
3.  Add the feature with the highest accuracy to the base model.

1.  If the accuracy improves, keep the feature in the base model.
2.  If the accuracy does not improve, discard the feature and move on to the next highest feature.

5.  Repeat step 3 until all features have been tested against the base model.

  

![](https://msoese.atlassian.net/wiki/download/attachments/853016632/image2019-5-5_15-34-2.png?version=1&modificationDate=1557088446092&cacheVersion=1&api=v2)

Figure 1: Greedy Algorithm

### Timeseries Data Extraction

The periodic sensor readings from the buoys produce timeseries with much variability over short periods of time. While considering the timeseries as a whole, conditions which accommodate algae blooms may stabilize up to a month in advance of an algae bloom occurrence. To account for this, we utilize a sliding window approach which shifts non-overlapping feature and target windows over the entire data set, extracting a specified percentile from each window. Where the extracted values from the feature windows serve as the data which will be used as inputs to the logistic regression model, while the extracted values from the target windows serve as the actual true values for model training and testing. This process reduces the noise between data points (Figure 2) and allows for predicting if a bloom will occur within a reasonable window of time.

Figure 2 depicts a 2-plot comparison of BGA RFU levels over the entire data set prior to (left) and after window extraction (right). This example shows how the BGA RFU levels are relatively smoothed by extracting the 95th  percentile (near max value) from a sliding 24-hour target window.

![](https://msoese.atlassian.net/wiki/download/attachments/853016632/image2019-5-5_13-57-53.png?version=1&modificationDate=1557082677071&cacheVersion=1&api=v2)

Figure 2: BGA data prior to window extraction (left), and BGA data after window extraction (right).

As an example, given a feature window size of 2 days, a target window size of 1 day, a shift amount of 1 day, and a dataset with a length of 6 days, our sliding window method would behave similar to what is depicted in Figure 3. In this example our process begins by creating a 2 day feature window, beginning at the start of the dataset, and a 1 day target window immediately following the feature window, spanning a total length of 3 days over the dataset. The windows then shift by 1 day, creating a new feature and target window over a period of time 1 day later than the first iteration. This process repeats until the entire dataset is covered, creating a total of 8 windows in this example. Single values are then extracted from each window (such as the mean), where a total of 4 feature values, and 4 associated target values will be used for training/testing models.

Figure 3 depicts our sliding window method used to extract data points from a subset series (or window) within the total timeseries. A min, mean, max, or any percentile value may be extracted from the given into a single data point. After extracting an input and target data point from the feature and target windows, each window is then shifted a given amount until the entire data set has been extracted.

![](https://msoese.atlassian.net/wiki/download/attachments/853016632/image2019-5-5_13-58-4.png?version=1&modificationDate=1557082688032&cacheVersion=1&api=v2)

Figure 3: Diagram of Sliding Window Method

  

Our sliding window method can incorporate variable separations in time, as depicted by the yellow delta segments in Figure 4. These deltas may be customized for specific features to narrow and fine tune optimal feature window size and placement or remain constant for the entire data set to effectively allow the model to predict at further points ahead in time.

![](https://msoese.atlassian.net/wiki/download/attachments/853016632/image2019-5-5_13-58-12.png?version=1&modificationDate=1557082695936&cacheVersion=1&api=v2)

Figure 4: Diagram of Sliding Window Method with Delta Period Between Features and Targets

  

Our sliding window method can incorporate variable feature window sizes for specific features as shown in Figure 5. Window sizes for specific features will remain constant throughout the data set (e.g. feature 1 in Figure 5) but can vary from other feature windows in length (e.g. feature 2 in Figure 5). This allows for custom feature window sizes within the same dataset. This provides the capability for optimizing window sizes on each feature as oppose to using a universal feature window size, since peak correlation periods can occur at different points in time from one feature to another.

![](https://msoese.atlassian.net/wiki/download/attachments/853016632/image2019-5-5_13-58-28.png?version=1&modificationDate=1557082711635&cacheVersion=1&api=v2)

Figure 5: Diagram of Sliding Window Method with Variable Window Sizes



