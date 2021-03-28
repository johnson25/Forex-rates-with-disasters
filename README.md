# Forex-rates-with-disasters
**Currency Exchange Rate Prognosticator**

![](RackMultipart20210224-4-blyloy_html_1780bbb92b112e.png) ![](RackMultipart20210224-4-blyloy_html_fa8fc6d855bf905b.png)

![](RackMultipart20210224-4-blyloy_html_268fea6b07464865.gif)

## _Artificial Intelligence_

## _Assignment_

![](RackMultipart20210224-4-blyloy_html_5bd8ba1a5ffe80ce.gif)

# **Currency Exchange Rate Prognosticator**

![](RackMultipart20210224-4-blyloy_html_b117a5828b4b6840.gif)

![](RackMultipart20210224-4-blyloy_html_f61ce47dd1372404.gif)

**TABLE OF CONTENTS:**

1. **Introduction**  **2**

  1. How natural disasters affect currency exchange market? 2
  2. Goal 3

1. **Background 3**
  1. Disaster 3
  2. Factors influencing forex rates 3
  3. Machine Learning Algorithms 5

1. **Datasets 6**
  1. Forex dataset 6
  2. Factors influencing forex rates 7
  3. Disaster dataset 8

1. **Implementation 9**
  1. Pre-Processing 9
  2. Exploratory Data Analysis 10
  3. Data Preparation 11

1. **Results 12**
  1. Linear Regression 12
  2. Decision Tree Regression 13
  3. XG Boost Regression 13
  4. KNN 14
  5. Support Vector regression 15
  6. Artificial Neural Network 15
  7. LSTM 16
  8. Comparison of algorithms 15
2. **Discussion 17**

1. **Conclusion 18**

1. **Future work 18**

1. **References 18**

1. **Appendix 20**

1. **INTRODUCTION:**

&quot; [When money talks, there are few interruptions.](https://www.inspiringquotes.us/quotes/lLHc_9EKTH8r3)&quot; - [Herbert V. Prochnow](https://www.inspiringquotes.us/author/7149-herbert-v-prochnow)

This world runs on money, and it dictates everyone&#39;s life in the modern-day world. What is money? Money or Currency is the run-of-the-mill for the exchange of goods and services you require in day-to day life. In the olden days, the exchange was conducted based on Barter system where the goods were bought in exchange of other goods. But later as Gold&#39;s prominence skyrocketed, the medium of exchange was controlled by Gold and other precious metals. This kind of currency is called &#39;Commodity money&#39;. It dealt major limitations of the bartering system effectively. Due to its scarcity and rapidly growing economies gold was not a viable option to conduct exchange further which gave birth to flat money currencies, the value of which was not determined by any commodity but was determined solely on the supply and demand. So, to be precise the stronger the economy of a country, stronger their currency value. So, purchasing power which is dictated by inflation rate is directly proportional to the currency value in the market.

Foreign exchange market, which is predominantly called &quot;FOREX market is the most colossal financial markets in the world. According to Statista, to anyone&#39;s surprise the average per day turnover in the global foreign exchange market in 2019 is nearly 6.6 trillion US dollars. Most of the transactions on the planet are carried out in US Dollars so USD is considered as the most influential currency. The currency value of a country comprehends well the economic health of that country. The currency value is determined by two methods one is Fixed rate which is solely defined by the Government or central bank. The other is floating rate which predominantly defined by the market events, supply and demand etc... Since the prediction of floating rate value is a significant task due to its unpredictability, there are several factors that influence the floating currency values. They are namely:

- Inflation
- Interest rates
- Gold prices
- Government debt
- Speculation
- Political stability and economic performance
- Trade
- Relative currencies

  1. **How natural disasters affect currency exchange market?**

Events that occur across the globe did have impact on the forex market all throughout the history. These events include political stability where during the election season there was fluctuation in the currency values. Due to recession, the currency values plummet. Wars or political instability might take hefty toll on the country&#39;s economy as well as its currency value.

Natural disasters leave out a catastrophic influence on the currency values. They directly affect the economy of the country by causing substantial damage to the property, manufacturing sector, human loss and morale etc., These events cripple the country and affect the business, supply and demand which in result weakens the currency. The affected country contributes from the GDP to repair the damages which slows down the economy. This might also lead to speculation where the investors speculate the demand for the currency might depreciate and stop doing business with it which substantially depreciates the value of the currency. According to the National Centers for Environmental information (NCEI), USA saw nearly 119 natural disaster events between 2010 -2019 which caused an average damage of worth $80.2 billion per year. Hence several empirical studies suggest that natural disasters have a negative impact on the nation&#39;s economy.

  1. **Goal:**

The primary goal of this project is to delineate the impact of the natural disasters on forex rate fluctuations. In addition to that, depicting how the factors such as gold prices, GDP, inflation, relative currencies etc., influence the forex rates. In this project, we will be employing several Machine learning algorithms and compare their accuracies in predicting the forex values during an active natural disaster.

1. **BACKGROUND:**
  1. **Disasters**

Due to deteriorating condition of the planet, the occurrences of natural disasters have increased year by year. To lessen the consequences of these disasters Government funds the country out of its GDP to repair the damages which could have been used for developmental and productive ventures. Considering the current coronavirus situation, many currencies were on decline due to uncertainty in the economic scenario. So, this project is about delineating the impact of natural disasters on the forex market. There are several types of natural disasters. Some are namely storms, snowstorms, floods, earthquakes, volcano eruptions, drought, hurricanes, tornadoes, diseases/ pandemics etc.,

Not every natural disaster affects the economy, but only certain disasters do. Natural disasters such as hurricanes, tornados, earthquakes, typhoons, fires etc., inflict hefty damages to the property and economy.

  1. **Factors influencing FOREX rates:**

Apart from our proposed idea of natural disasters affect the currency values, the aforementioned factors such as gold prices, GDP etc., are the main factors that influence the values.

Interest rates are highly correlated with the currency value. As the interest rates soar in the country, the local businesses will be impacted hugely as the consumer&#39;s purchasing power reduces and loan interest rates soar as well, which might jeopardize the confidence level of the investors.

GDP is directly proportional to the country&#39;s economic health. More the GDP the better the economy. GDP explains about country&#39;s exports, goods and services etc., So if the GDP is in good shape, it attracts more investors.

Inflation explains about the price of the goods and services available in the country. The exporting power of the country determines the inflation rate. The country with low inflation ate attracts more investors to invest money in the businesses in the country. Inflation is basically calculated using consumer price index. If the country&#39;s expenditure is greater than its income, then the country&#39;s economy runs into debts which might fail to attract new investments in the country harming the economic growth and falling of the currency value.

![](RackMultipart20210224-4-blyloy_html_d6633d7d96aa9b3.gif)

  1. **Machine Learning Algorithms:**

**Linear Regression:**

Linear regression is a statistical technique which depicts the relationship between a dependent variable and one or more independent variables. Model with one independent variable is called simple linear regression whereas the model with more than one independent variable is called multiple linear regression. Linear regression mathematical equation is of the form

where X is the independent variable and Y is the dependent variable. B is slope of the line and a is intercept.

**Decision Tree Regression:**

Decision tree algorithm builds regression in the form a tree. It splits the dataset into multiple smaller samples and fitted into a decision tree. This tree consists of decision nodes and leaf nodes. It works on the concept of top-down and greedy search across suitable branches.

**KNN Regression:**

K nearest neighbors algorithm works on predicting the numerical dependent variable based on distance functions or similarity functions. It is predominantly used in pattern recognition.

KNN computes the mean of the dependent variable of the K nearest neighbors. Euclidean and Manhattan distance functions are used for continuous variables.

**XG boost : **

Boosting is a process that uses a set of machine learning algorithms to combine weak learner to form strong learners in order to increase the accuracy of the model. In this boosting method, XG boost is one of the types that is designed to focus on computational speed and model efficiency.

**Support vector regression: **

This method is a discriminative classifier that is formally designed by a separative hyperplane.it is a representation of examples as points in space that are mapped so that the points of different categories are separated by gap as wide as possible.

**Artificial Neural Networks: **

ANNs are computing systems inspired by the biological neural networks that constitute animal brains. Such systems learn to do task by considering examples, generally without task-specific programming.

**LSTM: **

This is type of artificial neural network designed to recognize patterns in sequence of data, such as text, genomes, handwriting, the spoken word, or numerical time series data emanating from sensors, stock market and government agencies.

1. **DATASETS:**

For any Machine learning algorithm or data analysis, proper data set is the most fundamental part. There was not a single data set that was available on the internet to satisfy the problem statement. Data collection was one of the biggest challenges in this project. The project required to extract data on several features individually. The data used in this project is solely based on economic dynamics of United States of America. The Forex rates are against 1 USD (2000 -2019), the natural disaster data is the dataset enlisting the natural disasters that occurred in the USA (2000 – 2019)

![](RackMultipart20210224-4-blyloy_html_f0b3d06c44138df3.gif)

  1. **FOREX dataset:**

Downloaded the currency exchange rates data from BIS website and included only few powerful currencies in the market such as Euro, Pound Sterling, Australian Dollar, Canadian Dollar, New Zealand Dollar, Swiss Franc and Japanese Yen. The values under each currency feature are exchange rates in that currency for 1 US Dollar. So, for example first value under Euro is 0.991080 which means you can buy approximately 0.99 euro for 1 US Dollar.

![](RackMultipart20210224-4-blyloy_html_bf155facc6fecf36.gif)

  1. **Factors Influencing Forex rates:**

Downloaded the data related to factors such as Gold prices, debt, GDP, Consumer price index, producer price index from Federal Reserve Economic Data (FRED) operated by Federal Reserve Bank of St.Louis, USA. The data for each feature was extracted individually and features such as CPI and PPI are calculated monthly; GDP and debt are calculated quarterly, and gold prices are calculated daily. These all features were merged with the main dataset.

Gold prices – The daily historical data of gold prices in the USA

GDP – The gross domestic product of USA which is calculated quarterly.

Debt – The federal or Government debt in billions

CPI – Consumer Price Index, the measure of average change in prices of goods and services spent by urban consumers. CPI is used to calculate Inflation.

PPI – Producer Price Index, the measure of average change in prices of goods and services from the producer&#39;s perspective.

![](RackMultipart20210224-4-blyloy_html_18e628a107e00b80.gif)

  1. **Disaster dataset:**

Downloaded all-natural disaster dataset, those occurred in the USA from 2000 to 2020. It includes the disasters, the amount of damage they incurred and number of deaths

Name – name of the disaster

Disaster – the type of disaster (Flooding, Tropical Cyclone, Drought, Freeze, severe storm, winter storm, wildfire etc.,)

Begin Date – the date on which the natural disaster occurred

End date – the date on which the natural disaster subsided

Damage Cost – the amount of damage incurred by the disaster in millions of dollars

Deaths – Number of casualties due to the disaster

![](RackMultipart20210224-4-blyloy_html_492e8a6ad615cc96.gif)

1. **IMPLEMENTATION:**

Our project comes under supervised learning as we are predicting the currency values (dependent variable) by training the model on several features called predictor variables. The implementation phase of the project includes data integration, data wrangling, exploratory data analysis, data preparation accordingly to employ algorithms.

![](RackMultipart20210224-4-blyloy_html_585e1da157d0fece.gif)

  1. **Pre-Processing:**

After data collection, the data was integrated into one single data frame in the jupyter notebook. But it needed lot of prerequisite work to merge the three different datasets. The forex dataset has records for each day from 2000 to 2019. So, the whole dataset was supposed to be integrated to everyday level. The features CPI and PPI were monthly data, so they were replicated for the whole month to integrate them into daily level. GDP and debt features are quarterly data, so the values were replicated for four months to integrate into the forex dataset level. The natural disaster dataset has 265 records which means there were 265 natural disaster incidents between 2000 to 2019. The forex dataset and factors dataset were merged on the Date column where the Date column was transformed to Datetime format from object for no hassle merge. The natural disaster dataset was merged with the main dataset on begin date and inner join which will result in the dataset with no disaster where there is no record from disaster dataset for a specific date. Least significant variables are dropped based on their correlation with the forex rates and factors.

Handling Null Values:

There are numerous null values in the dataset. Most of them are due to no records or activity during the weekends as the forex market is closed on weekends. So, these types of null values are handled by dropping the records on weekends. The remaining null values in the currency columns are basically missing values in the records which were handled by using interpolate function with linear method. The missing values in the disaster group of data is basically due to no occurrence of disaster on a specific date so the null values in Disaster column are filled with &#39;No disaster&#39; string. Created a new column disaster\_event with values basically 0 and 1. If there is no occurrence of a disaster this category holds value 0 and if there is an occurrence of disaster it holds the value one. The missing values in Damage and Deaths columns are imputed with 0. Hence, our data was made ready for the exploratory data analysis.

  1. **Exploratory Data Analysis:**

Performing Exploratory data analysis helps us analyze the data and find the insights or patterns in the data. Plotted a graph of all the currencies and how they range over 19 years from 2000 – 2019. One of the interesting insights is after first quarter of 2008, there is huge drop in the currency values against USD due to the great recession globally. According to the theory, due to recession USD should depreciate as well but it did not because the investors considered USD as the &quot;safe haven&quot; and sold other currencies against the USD which resulted in shortage of USD available and appreciated its value.

The number of instances of each category of the disaster was plotted and it infers that there are 69 instances of severe storm, 22 of tropical cyclone, 17 of Flooding, 14 of drought, 11 of wildfire, 4 of winter storm, 3 of freeze.

Correlation heat matrix briefs about the variables which are correlated and many of the variables that were considered are strongly correlated with dependent variables.

All the plots that were made use of for data analysis are included in the Appendix section for your reference.

  1. **Data Preparation:**

Before the data is fed into the machine learning models, it must be transformed into the specific format so that the algorithm can interpret the data well.

- Firstly, the categorical variables are transformed into numerical values by employing label encoder. Each category of disaster column is transformed to numerical from categorial text type with label encoder where it assigns a unique number for each category.

- The date column is split into day, month, year, and week columns to have the date values in the dataset. The process was implemented using date functions such as dt.year, dt.week, dt.day, dt.month
- Lags for the predictor variables were introduced into the dataset to backward the data in the data set. As it is a time series model, it can be transformed to supervised machine learning using lags. Here we used shift 2 and the rows with null values are dropped from the datset.

The final step in the data preparation is splitting the data into train set and test set for the algorithms to train on the data and predict the values based on it. As it is a regression problem, we have predictor variables and dependent variable.

X (predictor variables) = (date, disaster data, gold prices, GDP, debt, PPI, CPI, all the relative currencies)

Y (dependent variable) = Currency you are predicting

We have the X and Y into x\_ train, x\_test, y\_train and y\_test respectively with the help of train\_test\_split method imported from sklearn.

For each algorithm we have created a function to make things easier. First block executes the splitting of the dataset. Second block defines the machine learning model. Then the model is fit into the x\_train and y\_train to train the model on the training data. Then the trained model is applied on the x\_test dataset and is modeled to predict the values of y for those provided x\_test values. The predicted values are stored in a dataframe along with the actual y\_test values and are compared on how well the algorithm has predicted the values.

For most of the algorithms, MinMax() scaler was used to standardize the values between 0 and 1 so that the data is fed uniformly into the model. For deep learning, SVR and KNN it is recommended.

1. **RESULTS:**

Euro, Pound Sterling, Australian Dollar, Canadian Dollar, Swiss Franc, Japanese Yen and New Zealand are the currencies that were predicted based on the factors and the natural disasters. To make it concise, the results and evaluation will be presented on each algorithm&#39;s performance on predicting Euro value. For other currencies please refer Appendix as well as code.

Our problem statement is to predict a currency value when the disaster event occurred. In other words, predict the currency value when disaster\_event = 1.To represent how well the model has predicted the currency values, a line plot and a bar plot were plotted on the actual values and predicted values. The closer the bars or lines to each other the better the prediction rate.

The performance of the model was evaluated using generic evaluation techniques including Root mean square error, mean squared error, mean absolute error, Mean absolute percentage error, and accuracy. As the fluctuations in the currency values are basically very little, mean absolute percentage error (MAPE) will be more significant to determine the performance when compared to other evaluation techniques.

**Mean absolute percentage error** = (abs ((y\_test - y\_pred) / y\_test).mean()) \* 100

  1. **Linear Regression:**

Sklearn library offers linear regression function called LinearRegression() which was used to perform simple linear regression on the dataset. From the figure we understand that the Actual and predicted values are lined up so closely with mean absolute percentage error of 0.288%

![](RackMultipart20210224-4-blyloy_html_a582a23e626b6a1f.gif)

Mean Absolute Percentage Error: 0.2883427251277821

  1. **Decision Tree Regression:**

Sklearn library offers linear regression function called DecisionTreeRegressor() which was used to perform decision tree regression on the dataset. The parameters criterion was initialized to &#39;mse&#39; which is mean squared error and the depths included were 10. From the figure we understand that the Actual and predicted values are lined up so closely with mean absolute percentage error of 0.615%

![](RackMultipart20210224-4-blyloy_html_9379d5160d9b0b38.gif)

Mean Absolute Percentage Error: 0.6150211779928688

  1. **XG Boost Regression:**

Xgboost.sklearn library offers linear regression function called XGBRegressor () which was used to perform XG boost regression on the dataset. The parameters in the regressor include

(objective =&#39;reg:squarederror&#39;, seed=100, n\_estimators=100, max\_depth=3, learning\_rate=0.1, min\_child\_weight=1, subsample=1, colsample\_bytree=1, colsample\_bylevel=1, gamma=0)

From the figure we understand that the Actual and predicted values are lined up so closely with mean absolute percentage error of 0.572%

![](RackMultipart20210224-4-blyloy_html_78c8150eade2e103.gif)

Mean Absolute Percentage Error: 0.5722705803583533

  1. **KNN Regression:**

Sklearn library offers linear regression function called neighbors.KNeighborsRegressor(), (with number of neighbors k =4)which was used to perform simple linear regression on the dataset. From the figure we understand that the Actual and predicted values are lined up so closely with mean absolute percentage error of 0.846%

![](RackMultipart20210224-4-blyloy_html_38541a8c45c4bb52.gif)

Mean Absolute Percentage Error: 0.8461238567519557

  1. **Support Vector regression:**

Sklearn library offers support vector regression function called SVR (). It needs to pass a parameter called kernel where we initialized it to &#39;poly&#39;which was used to perform simple polynomial regression on the dataset. From the figure we understand that the Actual and predicted values are lined up so closely with mean absolute percentage error of 3.778%

![](RackMultipart20210224-4-blyloy_html_e1d4f10274e5b2da.gif)

Mean Absolute Percentage Error: 3.7780422840140773

  1. **Artificial Neural Network:**

Deep Learning is one of the powerful techniques in the field of machine learning. Deep learning is the imitation of how human naturally thinks.

One of the prominent algorithms in the field of deep learning is Artificial Neural Network which is predominantly used in regression analysis.

Just like human brain, ANN consists of numerous interconnected fundamental units called Neurons. These Neurons transmit the data across the Network. These Neurons also called perceptron. ANN makes use of activation function to convert incoming signals to outgoing signals. These incoming signals is used as input for the network.

In this scenario we have 30 units in input layer input layer. And we use relu as activation function. In the hidden layer we have 15 units and activation layer is relu.

The output layer has a single unit only.

We use the mean squared error as loss function and adam optimizer to reduce the loss and optimize the algorithm.

![](RackMultipart20210224-4-blyloy_html_66c4ab8a78d29d07.gif)

Mean Absolute Percentage Error: 16.35953568232143

  1. **LSTM:**

RNN&#39;s main purpose of implementation is to analyze and predict the data in sequential manner.

A recurrent neural network is a neural network where the output of the network from one-time step is provided as an input in the subsequent time step. This allows the model to decide as to what to predict based on both the input for the current time step and direct knowledge of what was output in the prior time step.

In RNN the output from a step of a network is fed as the input to the adjacent step. This process enables the algorithm in decision making of the prediction on the basis of the input of current step and the memory of the output from the previous step.

One of the effective RNN algorithms is LSTM which stands for Long short-term memory. It is successful as it enervates the limitations in training a recurrent RNN. LSTMs have an internal memory that operates like a local variable, allowing them to accumulate state over the input sequence. An LSTM model expects data to have the shape:

[samples, timesteps, no. of input features]

Therefore, the shape of our data was changed to :

Train set: (3913, 1, 39)

Test set:(1305, 1, 39)

Developed a model with a single hidden LSTM layer with 30 units. The LSTM layer is followed by a fully connected layer with 30 nodes that will interpret the features learned by the LSTM layer. Finally, an output layer will predict the test data.

We use the mean squared error as loss function and adam optimizer to reduce the loss and optimize the algorithm.

![](RackMultipart20210224-4-blyloy_html_63d336ca4e7b95f7.gif)

Mean Absolute Percentage Error: 14.707605277468655

1. **DISCUSSION:**

The models employed in this project have done a decent job on predicting the currency values based on the factors and the disaster data. One clear observation from the above analysis is natural disasters do not directly impact the currency values but instead affect GDP, Inflation rate, CPI etc., which directly fluctuate the currency values. And also, the change in the currency value also depends on several other factors such as type of the disaster event, degree of severity, purchasing activity, political scenario like elections etc., Still it is quiet a mystery on how natural disaster directly impact the currency value.

1. **CONCLUSION:**

Our main objective was to predict the currency values based on natural disasters. From the above documented results, it is comprehensible that Linear regression, KNN regression, XG boost regression and decision tree regression have less than 1% mean absolute percentage error when compared to the Support Vector regression, deep learning algorithms ANN and LSTM.

1. **FUTURE WORK:**

Researching more on how natural disasters directly affect the currency rates and train on more historical data probably since 1900s. Include several other factors such as shares, political scenario, fuel prices, news headlines which might jeopardize investors in buying the currency.

1. **REFERENCES:**

Gan, W.-S. and Ng, K.-H. (1995). Multivariate forex forecasting using artificial neural networks, Neural Networks, 1995. Proceedings., IEEE International Conference on, Vol. 2, pp. 1018{1022 vol.2.

[https://www.bis.org/statistics/xrusd.htm](https://www.bis.org/statistics/xrusd.htm)

https://fred.stlouisfed.org

https://www.compareremit.com/money-transfer-guide/key-factors-affecting-currency-exchange-rates/

[https://www.investopedia.com/terms/p/ppi.asp](https://www.investopedia.com/terms/p/ppi.asp)

[https://www.climate.gov/news-features/blogs/beyond-data/2010-2019-landmark-decade-us-billion-dollar-weather-and-climate](https://www.climate.gov/news-features/blogs/beyond-data/2010-2019-landmark-decade-us-billion-dollar-weather-and-climate)

https://www.odi.org/publications/5011-economic-and-financial-impacts-natural-disasters-assessment-their-effects-and-options-mitigation

[https://www.nottinghampost.com/special-features/how-exactly-forex-market-impacted-3656159](https://www.nottinghampost.com/special-features/how-exactly-forex-market-impacted-3656159)

https://towardsdatascience.com/metrics-and-python-850b60710e0c

https://xgboost.readthedocs.io/en/latest/python/python\_api.html

[https://www.learntotrade.co.uk/factors-affecting-exchange-rates/](https://www.learntotrade.co.uk/factors-affecting-exchange-rates/)

[https://scikit-learn.org/stable/supervised\_learning.html#supervised-learning](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)

[https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)

[https://www.google.com/url?sa=t&amp;source=web&amp;rct=j&amp;url=https://towardsdatascience.com/introduction-to-artificial-neural-networks-ann-1aea15775ef9&amp;ved=2ahUKEwjo1OLNgY3qAhVQTBUIHXG5D8MQFjAiegQICBAB&amp;usg=AOvVaw0xHqOf6yoecIB-9ezajq2u](https://www.google.com/url?sa=t&amp;source=web&amp;rct=j&amp;url=https://towardsdatascience.com/introduction-to-artificial-neural-networks-ann-1aea15775ef9&amp;ved=2ahUKEwjo1OLNgY3qAhVQTBUIHXG5D8MQFjAiegQICBAB&amp;usg=AOvVaw0xHqOf6yoecIB-9ezajq2u)

**APPENDIX:**

Comparison of currencies against USD = 1:

![](RackMultipart20210224-4-blyloy_html_e8a14efaf4d42110.gif)

Number of instances of each disaster type in the past 19 years:

![](RackMultipart20210224-4-blyloy_html_342d975c908ea949.gif)

13 **|** Page
