# Humanitarian Unrest Classifier
## Project McNulty: Team Saving the World
Welcome to Project McNulty. The motivation behind this project is to identify areas in the world that may be susceptible to humanitarian unrest, specifically focusing on crises that might prompt a mass migration of refugees.

Our team will approach the question by seeking to individually create machine learning models within identified key areas that may lead to humanitarian unrest: economic, environmental and climatic, terrorism and national security, political and population demographics, and food. We will use PCA and other pre-processing methods, choose an appropriate model, and use the output of individual predictions as features for a final model.

Our final model will use pre-processing methods to ensure that the results of our individual models are comparable. We will then create an index using a classification algorithm to assign countries a value of 'immediate risk (in crisis)', 'high risk', 'moderate risk', and 'low risk'.

Note: We are attempting to share our data on a single postgres database on one ec2 cluster. Currently we are in the process of developing safe and secure access to this pipeline in a way that would allow us to share the data on the ec2 machine while using the computing power of our local machines to perform the analyses.

#### Economic ([Bryan Bumgardner](https://github.com/BryanBumgardner))
There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text.

#### Environmental ([Daniel Yawitz](https://github.com/yawitzd))
There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text.

#### Terrorism ([Kenneth Chadwick](https://github.com/outsideken))
There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text.

#### Political ([Ken Myers](https://github.com/kennmyers))
There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text.

#### Food ([Emily Hough-Kovacs](https://github.com/emilyhoughkovacs/))
The data for the food algorithm comes primarily from the World Food Programme's [global food prices database](https://data.hdx.rwlabs.org/dataset/wfp-food-prices/resource/b5b850a5-76da-4c33-a410-fd447deac042). This 580,000 row database contains monthly price information on many food prices in markets across the globe. The advantage of this data is that it is precise to the city level (country, region and city) as well as to the month level. The prediction algorithm will be chosen by what best optimizes time-series at that level. Some foreseeable roadblocks include prices that may be reported in local currencies. Another consideration is considering the purchasing power of a citizen in each region. Further exploratory analysis may be required.
