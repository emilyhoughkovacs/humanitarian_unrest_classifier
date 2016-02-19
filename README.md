# Humanitarian Unrest Classifier
## Project McNulty: Team Saving the World
Welcome to Project McNulty. Our goal is to identify areas in the world that may be susceptible to humanitarian unrest, specifically focusing on crises that might prompt a mass migration of refugees.

Our team will create machine learning models to examine key factors that may lead to humanitarian unrest: economic, environmental and climatic, terrorism and national security, political and population demographics, and food. We will use PCA and other pre-processing methods, choose an appropriate model, and use the output of individual predictions as features for a final model.

The final model will use pre-processing methods to ensure the results of our individual models are comparable. We will then create an index using a classification algorithm to assign countries a value of 'immediate risk (in crisis)', 'high risk', 'moderate risk', and 'low risk'.

This project will be beneficial to anyone with interest in geopolitics. Both firms and individuals with international interests could use this graph to make educated business, travel, and life decisions. We hope to package the project as an easy-to-navigate interactive infographic.

We are sharing our data on a single database on one ec2 server. We are using Postico, a graphical interface, to locally connect to this shared postgres database.

#### Economic ([Bryan Bumgardner](https://github.com/BryanBumgardner))
The idea is simple: if you are in poverty, you aren't happy about it. Also, if something bad happens (war, natural disaster, change in leadership) poverty exacerbates the situation. The World Bank keeps detailed records of various economic factors that can contribute to poverty. The UN Refugee Agency’s (UNHCR) eighth annual High Commissioner’s Dialogue on Protection Challenges described several of these factors: high unemployment, especially among youth, uneven development, lack of access to international markets, and income inequality. To measure these horrible things, I'm looking at some data recorded by the World Bank, and using these measures as features:
- The Gini Coefficient: the most commonly used way of measuring wealth distribution. The larger the wealth gap, the most restless the poorer residents.
- Unemployment of youth, and also general unemployment.
- GDP per capita, normalized by the current PPP (international dollar).
- GDP per capita growth from the previous year for the bottom 40% of the population.
- Food imports as a percentage.
- Total labor force participation. 
With these economic factors accounted for, our model will get a clearer look at the health of a country on some specific measures. 

#### Environmental ([Daniel Yawitz](https://github.com/yawitzd))
Data for the environmental algorithm comes primarily from the International Disasters Database (aka [EM-DAT](http://www.emdat.be/)), which contains a record of 7000 natural disasters since 1980, and the number of people displaced or affected. That data will be tied to a record of all major earthquakes (magnitude 5.3+) since 1988 from the United States Geological Survey ([USGS](http://earthquake.usgs.gov/earthquakes/search/)), and other data that measures the severity of other disasters (floods, storms, droughts, etc). The final model will use the severity of an event and other economic and political data to predict whether or not the disaster will displace a significant number of people.

Foreseeable challenges include: tying the earthquake records (each event is a lat-long) to its EM_DAT record (by country), finding severity data for other types of disasters, and selecting enough features to not over-generalize this global model.

#### Security & Terrorism ([Kenneth Chadwick](https://github.com/outsideken))
This feature is developed from a number of indices developed by thinktanks and nonprofits such as the Institute for Economics and Peace (IEP) and the Council on Foreign Releations (CFR) that cover acts of terrorism, insurgencies, and border disputes.  IEP's Global Peace Index (GPI) is also included in this feature.  The data is sourced from a wide range of respected sources, including the International Institute of Strategic Studies, The World Bank, various UN Agencies, peace institutes and the EIU.  Each of these indices are combined in a weighted model to produce an overall Security & Terrorism score.

Sources:
* Council on Foreign Relations (CFR), [Invisible Armies Insugency Tracker](http://www.cfr.org/wars-and-warfare/invisible-armies-insurgency-tracker/p29917)
* Institute for Economics and Peace (IEP), [Global Peace Index 2015](http://economicsandpeace.org/wp-content/uploads/2015/06/Global-Peace-Index-Report-2015_0.pdf),  [Global Terror Index 2015](http://economicsandpeace.org/wp-content/uploads/2015/11/Global-Terrorism-Index-2015.pdf)
* Uppsala Universitet, Depaterment of Peace and Conflict Research, Conflict Data Program - [UCDP/PRIO Armed Conflict Dataset v.4-2015, 1946 – 2014](http://www.pcr.uu.se/research/ucdp/datasets/ucdp_prio_armed_conflict_dataset/)

#### Political and Demographics ([Ken Myers](https://github.com/kennmyers))
The following features were examined in my model:
###### Politics

* Corruption Perception Index<sup>1</sup>
* Civil Liberties<sup>2</sup>
* Political Rights<sup>2</sup>
* Freedom Status<sup>2</sup>
* Ratio of Female Legislators<sup>3</sup>

###### General Demographics

* Gender Ratio<sup>3</sup>
* Population Growth<sup>3</sup>
* Age<5 Mortality<sup>3</sup>
* Life Expectancy<sup>3</sup>
* Population age 0-14<sup>3</sup>
* Population age 15-64<sup>3</sup>
* Population Age 65+<sup>3</sup>

1. Since 1995, Transparency International has kept a Corruption Perception Index on countries around the world. [According to their website](http://www.transparency.org/research/cpi/overview), this index's purpose is to 'score countries on how corrupt their public sectors are seen to be.' Fortunately for us, someone else has already compiled [all of this data into a single file](https://github.com/datasets/corruption-perceptions-index). The biggest challenge with this data is that after 2010 their method of indexing changed. The most likely options are to either exclude the latest data or to normalize the scores by each year.

2. This political data will be combined with data from [Freedom House](https://freedomhouse.org/report/freedom-world/freedom-world-2016) which keeps an index on the political rights, civil liberties, freedom status of countries around the world.

3. Age and gender demographics data from [The World Bank](http://data.worldbank.org/indicator) will also be analyzed. I used the dataset available from [Kaggle](https://www.kaggle.com/worldbank/world-development-indicators) which is slightly transformed from the original data. From these sets of data we will try to see if it is possible to predict a country's risk of a humanitarian crisis by their corruption perception and the their population's demographic. Such as, if a population has a higher percentage of males or a higher percentage of younger people, are they more likely to enter a crisis when they perceive their country as being corrupt.

#### Food ([Emily Hough-Kovacs](https://github.com/emilyhoughkovacs/))
The data for the food algorithm comes primarily from the World Food Programme's [global food prices database](https://data.hdx.rwlabs.org/dataset/wfp-food-prices/resource/b5b850a5-76da-4c33-a410-fd447deac042). This 580,000 row database contains monthly price information on many food prices in markets across the globe. The advantage of this data is that it is precise to the city level (country, region and city) as well as to the month level. The prediction algorithm will be chosen by what best optimizes time-series at that level. Some foreseeable roadblocks include prices that may be reported in local currencies. Another consideration is considering the purchasing power of a citizen in each region. Further exploratory analysis may be required.

#### Additional Background Reading on Internally Displaced Persons and Refugees:
* [Internal Displacement Monitoring Centre (IDMC)](http://www.internal-displacement.org/)
* [United Nations High Commission on Refugees](http://www.unhcr.org/cgi-bin/texis/vtx/home)
* [Uppsala Universitet, Depaterment of Peace and Conflict Research, Conflict Data Program](http://www.pcr.uu.se/research/ucdp/)
