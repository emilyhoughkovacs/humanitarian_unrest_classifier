# Humanitarian Unrest Classifier
## Project McNulty: Team Saving the World
Welcome to Project McNulty. Our goal is to identify areas in the world that may be susceptible to humanitarian unrest, specifically focusing on crises that might prompt a mass migration of refugees.

Our team will create machine learning models to examine key factors that may lead to humanitarian unrest: economic, environmental and climatic, terrorism and national security, political and population demographics, and food. We will use PCA and other pre-processing methods, choose an appropriate model, and use the output of individual predictions as features for a final model.

The final model will use pre-processing methods to ensure the results of our individual models are comparable. We will then create an index using a classification algorithm to assign countries a value of 'immediate risk (in crisis)', 'high risk', 'moderate risk', and 'low risk'.

This project will be beneficial to anyone with interest in geopolitics. Both firms and individuals with international interests could use this graph to make educated business, travel, and life decisions. We hope to package the project as an easy-to-navigate interactive infographic.

Note: We are attempting to share our data on a single postgres database on one ec2 cluster. Currently we are developing safe and secure access to this pipeline in a way that would allow us to share the data on the ec2 machine while using the computing power of our local machines to perform the analyses.

#### Economic ([Bryan Bumgardner](https://github.com/BryanBumgardner))
The idea is simple: if you are in poverty, you aren't happy about it. Also, if something bad happens (war, natural disaster, change in leadership) poverty exacerbates the situation. The World Bank keeps detailed records of something known as Gross Domestic Product Price Per Parity, which creates a GDP that adjusts for exchange rate across currencies. Logically, countries with worse GDP have worse political stability, bigger challenges coping with natural disasters, and lower quality of life. Using GDP PPP and tracing how it changes through time, I can apply this to the other data and identify countries that struggle to cope with crisis-causing factors. 

#### Environmental ([Daniel Yawitz](https://github.com/yawitzd))
Data for the environmental algorithm comes primarily from the International Disasters Database (aka [EM-DAT](http://www.emdat.be/)), which contains a record of 7000 natural disasters since 1980, and the number of people displaced or affected. That data will be tied to a record of all major earthquakes (magnitude 5.3+) since 1988 from the United States Geological Survey ([USGS](http://earthquake.usgs.gov/earthquakes/search/)), and other data that measures the severity of other disasters (floods, storms, droughts, etc). The final model will use the severity of an event and other economic and political data to predict whether or not the disaster will displace a significant number of people.

Foreseeable challenges include: tying the earthquake records (each event is a lat-long) to its EM_DAT record (by country), finding severity data for other types of disasters, and selecting enough features to not over-generalize this global model. 

#### Terrorism ([Kenneth Chadwick](https://github.com/outsideken))
There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text.

#### Political ([Ken Myers](https://github.com/kennmyers))
There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text.

#### Food ([Emily Hough-Kovacs](https://github.com/emilyhoughkovacs/))
The data for the food algorithm comes primarily from the World Food Programme's [global food prices database](https://data.hdx.rwlabs.org/dataset/wfp-food-prices/resource/b5b850a5-76da-4c33-a410-fd447deac042). This 580,000 row database contains monthly price information on many food prices in markets across the globe. The advantage of this data is that it is precise to the city level (country, region and city) as well as to the month level. The prediction algorithm will be chosen by what best optimizes time-series at that level. Some foreseeable roadblocks include prices that may be reported in local currencies. Another consideration is considering the purchasing power of a citizen in each region. Further exploratory analysis may be required.
