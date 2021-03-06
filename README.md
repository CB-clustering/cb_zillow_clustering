# Clustering 

## This file/repo contains information related to our clustering project, using thee Zillow dataset from the Codeup database.

## Project Description

This Jupyter Notebook and presentation explore the Zillow dataset from the Codeup database. The data used relates to 2017 real estate transactions relating to single family homes in three California counties, and different aspects of the properties. An important aspect of the Zillow business model is to be able to publish accurate home values; we intend to build a machine learning model that makes predictions about the log error of the homes in question.

We will use Residual Mean Square Error as our metric for evaluation; many models will be built using different features and hyperparameters to find the model of best fit.  One of the final deliverables will be the RMSE value resulting from our best model, contrasted with the baseline RMSE.

Additionally, a Jupyter Notebook with our main findings and conclusions will be a key deliverable; many .py files will exist as a back-up to the main Notebook (think "under-the-hood" coding that will facilitate the presentation).


## The Plan

The intention of this project is to follow the data science pipeline by acquiring and wrangling the relevant information from the Codeup database using a MySQL query; manipulating the data to a form suitable to the exploration of variables and machine learning models to be applied; and to graphically present the most outstanding findings, including actionable conclusions, along the way.

## Project Goals

The ultimate goal of this project is to build a model that predicts the log error of the homes in question with a higher accuracy than the baseline we have chosen--the purpose being to be able to produce better, more accurate log error predictions than Zillow's competitors. 

## Initial Questions

- Is there a time period that has a higher or lower log error?  

- What about a relationship between tax_value and logerror? (also: are these related targets?)

- Does the condition of the home have an impact on the logerror?

- Does (or...how does?) logerror differ by county?

- Does our whole theory about half bathrooms bear out??


##  Steps to Reproduce

In  the case of this project, there are several python files that can be used to acquire, clean, prepare and otherwise manipulate the data in advance of exploration, feature selection, and modeling (listed below).

We split the data into X_train and y_train data sets for much of the exploration and modelling, and was careful that no features were directly dependent on the target variable (tax_value).  we created a couple of features which produced some useful insights, and dropped rows with null values (our final dataset was 39574 rows long, from 52,442 that were downloaded using SQL)

Once the data is correctly prepared, it can be run through the sklearn preprocessing feature for polynomial regressiong and fit on the scaled X_train dataset, using only those features indicated from the recursive polynomial engineering feature selector (also an sklearn function).  This provided us with the best results for the purposes of this project.

LIST OF MODULES USED IN THE PROJECT, FOUND IN THE PROJECT DIRECTORY:
-- wrangle.py: for acquiring, cleaning, encoding, splitting and scaling the data.  
-- viz.py: used for creating several graphics for my final presentation
-- model.py: many, many different versions of the data were used in different feature selection and modeling algorithms; this module is helpful for splitting them up neatly.
-- feature_engineering.py: contains functions to help choose the 'best' features using certain sklearn functions 
-- cluster_model.py : contains functions to help Perform Clustering using kmeans libraries

 




## Data Dictionary

 

| Variable          | Description                            |Data types|
| ----------------- | -------------------------------------- |----------|
|parcelid           | Identifies the parcel ID  number       |int64     |
|bathrooms          | Indicates the number of bathrooms      |float64   |
|bedrooms           | Indicates the number of bedrooms       |float64   |
|condition          | Indicates the condition of the Property|float64   |
|sq_ft              | Calculated square footage of the home  |float64   |
|full_baths         | Indicates the number of full baths     |float64   |
|tax_value          |The estimated value of the home         |float64   |
|lot_size           |The size of the Property                |float64   |
|tax_amount         |taxes payed the previous year           |float64   |
|logerror           |Error log from previous predictions     |float64   |
|county             |Indicates the county location           |object    |
|structure_value    |The estimated value of the strucrure    |float64   |
|age                |Computed age  from the year built       |float64   |
|sq_ft_per_bathroom |Square footage per Bathrooms            |float64   |
|sq_ft_per_bedroom  |Square footage per bedroom              |float64   |
|sq_ft_per_room     |Square footage per room                 |float64   |
|has_half_bath      |Whether the property has half baths     |int64     |
|age_bin            |Age range of the property               |category  |
|tax_rate           |Calculated tax rate                     |float64   |
|price_per_sq_ft    |Calculated price per squarefoot         |float64   |
|Los_Angeles        |Belongs to Los Angeles County           |uint8     |
|Orange             |Belongs to Orange County                |uint8     |
|Ventura            |Belongs to Ventura County               |uint8     |

Variables created in the notebook (explanation where it helps for clarity):

- zillow_sql_query (The original query that is copied to 'zillow' DataFrame for manipulation and analysis)
- zillow (A copy of the above DataFrame for use in the notebook)
- train
- validate
- test
- X_train
- y_train
- X_validate
- y_validate
- X_test
- y_test
- train_scaled
- X_train_scaled
- y_train_scaled
- validate_scaled
- X_validate_scaled
- y_validate_scaled
- test_scaled
- X_test_scaled
- y_test_scaled
- X_train_kbest (scaled dataframe with only KBest features)
- X_validate_kbest (scaled dataframe with only KBest features)
- X_test_kbest (scaled dataframe with only KBest features)
- X_train_rfe (scaled dataframe with only RFE features)
- X_validate_rfe (scaled dataframe with only RFE features)
- X_test_rfe (scaled dataframe with only RFE features)

Missing values: there were only something around 200 missing values in the data; thus, we have dropped them in the wrangle.py file due to their relative scarcity.  By removing outliers, several thousand rows were dropped.

## Key findings, recommendations and takeaways
At the beginning of my study, I took a few high-level type questions to get a handle on the problem, and refined them slowly with the time that was available:

Is there a time period that has a higher or lower log error?

What about a relationship between tax_value and logerror? (also: are these related targets?)

Does the condition of the home have an impact on the logerror?

Does (or...how does?) logerror differ by county?

Does my whole theory about half bathrooms bear out?

Most of these initial questions were disproved through exploration and statistical testing, but there were a couple of gems, and the "condition" of a home was in fact one of the final attributes to make it into the "winning" model.

Our model, which has room for improvement, was capable of predicting the logerror of homes with a 1.5 percent improvement over baseline on the test data, and with similar margins for the train and validate data. We expect it to perform as well on unseen data as well.


 
## Recommendations
Building models that separate the counties might see some benefit, seeing as there area some different ways that the features work on logerror in different counties.

Explore locations/city ids--we are of the belief that this attribute and related ones (latitue and longitude, for example), may yield some insight.



## Next steps
The following is a brief list of items that we'd like to add to the model:

A cluster relating to other aspects of a home's value (structure, land, etc)
Also a cluster related to physical aspects of a home (beds and baths, for instance)
Continue exploring other features' relationship to logerror, with an eye to mathmatical relationships (our early exploration mistakenly went down a dead end or two because of a failure to switch gears from the target variable of the previous project that this one builds off of)
 




