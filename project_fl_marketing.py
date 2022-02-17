## Analysis of customer data to create Buyer Personas' using K-means clustering.

from unicodedata import numeric
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st
from PIL import Image

@st.cache(allow_output_mutation=True)
def get_data(url):
    marketing_data = pd.read_csv(url, parse_dates = ['Dt_Customer'])
    return marketing_data

@st.cache
def get_downloadable_data(df):
    return df.to_csv().encode('utf-8')


url = 'https://raw.githubusercontent.com/federicoleo/data_science_project/master/marketing_data.csv'
marketing_data_static = get_data(url)
marketing_data = marketing_data_static.copy()

header = st.container()
data_exploration_cleaning = st.container()
data_visualization = st.container()
modeling = st.container()


with header:
  st.title('Analysis of customer data to create Buyer Personas using K-means clustering.')
  st.write('Dataset source: [click link](' + url + ')')
  st.download_button('DOWNLOAD RAW DATA', get_downloadable_data(marketing_data_static), file_name='marketing_raw.csv')

with data_exploration_cleaning:
  st.markdown('''
  ## 1. Data Exploration and Cleaning.

  #### This is how our data are represented in the DataFrame:''')
  st.dataframe(marketing_data)

  st.write('Our Dataframe has', marketing_data.shape[0],'rows and', marketing_data.shape[1], 'columns.')
  st.write(marketing_data.columns.T)

  column_expl = st.button('Click for full column description')
  if column_expl: 
    st.markdown('''
    - ID=Customer's unique identifier
    - Year_Birth=Customer's birth year
    - Education=Customer's education level
    - Marital_Status=Customer's marital status
    - Income=Customer's yearly household income
    - Kidhome=Number of children in customer's household
    - Teenhome=Number of teenagers in customer's household
    - Dt_Customer=Date of customer's enrollment with the company
    - Recency=Number of days since customer's last purchase
    - MntWines=Amount spent on wine in the last 2 years
    - MntFruits=Amount spent on fruits in the last 2 years
    - MntMeatProducts=Amount spent on meat in the last 2 years
    - MntFishProducts=Amount spent on fish in the last 2 years
    - MntSweetProducts=Amount spent on sweets in the last 2 years
    - MntGoldProds=Amount spent on gold in the last 2 years
    - NumDealsPurchases=Number of purchases made with a discount
    - NumWebPurchases=Number of purchases made through the company's web site
    - NumCatalogPurchases=Number of purchases made using a catalogue
    - NumStorePurchases=Number of purchases made directly in stores
    - NumWebVisitsMonth=Number of visits to company's web site in the last month
    - AcceptedCmp1=1 if customer accepted the offer in the 1st campaign, 0 otherwise
    - AcceptedCmp2=1 if customer accepted the offer in the 2nd campaign, 0 otherwise
    - AcceptedCmp3=1 if customer accepted the offer in the 3rd campaign, 0 otherwise
    - AcceptedCmp4=1 if customer accepted the offer in the 4th campaign, 0 otherwise
    - AcceptedCmp5=1 if customer accepted the offer in the 5th campaign, 0 otherwise
    - Response=1 if customer accepted the offer in the last campaign, 0 otherwise
    - Complain=1 if customer complained in the last 2 years, 0 otherwise
    - Country=Customer's location
    ''')

  #Income column contains some null values. We are going to drop them operate on the rest of the data."""

  marketing_data.dropna(inplace=True)

  #We noticed also that the Income column contained some blank spaces in the title and also the data type string with the $ symbol. We need to convert the column into int dtype in order to let us perform analysis also on this column.

  marketing_data.columns = marketing_data.columns.str.replace(' ', '')

  marketing_data['Income'] = marketing_data['Income'].str.replace("[\$\,]", '').astype(float)

  marketing_data['Income'] = marketing_data['Income'].astype(int)

 
  outliers_df = marketing_data.drop(['ID', 'AcceptedCmp1', 'Dt_Customer', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp2', 'Response', 'Complain', 'Country', 'Education', 'Marital_Status'], axis = 1)

  outliers_df.head()

  outliers_df.plot(subplots=True, layout=(4,4), kind='box', figsize=(15, 17))
  plt.suptitle('Outliers detection', fontsize=15, y=0.9)

  #Removing outliers
  marketing_data = marketing_data[marketing_data['Year_Birth'] > 1939].reset_index()
  marketing_data = marketing_data[marketing_data['Income'] < 666666]


  marketing_data['Tot_amount'] = marketing_data['MntWines'] + marketing_data['MntFruits'] + marketing_data['MntMeatProducts'] + marketing_data['MntFishProducts'] + marketing_data['MntSweetProducts'] + marketing_data['MntGoldProds']

  marketing_data['Tot_purchases'] = marketing_data['NumStorePurchases'] + marketing_data['NumCatalogPurchases'] + marketing_data['NumWebPurchases']
  #we excluded the NumDealsPurchsed because it could result in a double counting for orders because it is not specified whether e.g. a WebPurchase could have been made with a deal.

  marketing_data['Average amount'] = marketing_data['Tot_amount'] / marketing_data['Tot_purchases']

  marketing_data[['Tot_amount', 'Tot_purchases', 'Average amount']].describe()

  marketing_data['Tot_purchases'].value_counts().sort_index(ascending=True)

  marketing_data = marketing_data[marketing_data['Tot_purchases'] != 0]

  marketing_data.describe()

  st.subheader('Added features:')
  st.dataframe(marketing_data[['Tot_amount', 'Tot_purchases', 'Average amount']].describe())

  st.subheader('Average customer look-a-like:')
  average_customer = marketing_data.describe().loc['mean']

  average_customer

  st.subheader('Conversion rate for each campaign')
  st.write('Conversion rate = the percentage of customers who actually made a purchase thanks to the specific campaign out of all the customers involved.')
  
  #Conversion rate for each campaign

  def conversion_rate_func(column):
    total = marketing_data['ID'].nunique()
    subscribers = marketing_data[marketing_data[column]== 1]['ID'].nunique()
    conversion_rate = subscribers / total * 100
    return str(round(conversion_rate, 1))+' %'

  campaigns_df = marketing_data[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']]


  for x in campaigns_df:
    count = range(len(campaigns_df))
    print('The conversion rate for campaign {} is'.format(x[-1]), conversion_rate_func(x))

  col1, col2, col3, col4, col5, col6 = st.columns(6)
  col1.metric("Campaign 1", "6.4%")
  col2.metric("Campaign 2", "1.4%")
  col3.metric("Campaign 3", "7.4%")
  col4.metric("Campaign 4", "7.4%")
  col5.metric("Campaign 5", "7.3%")
  col6.metric("Campaign 6", "15.1%")

  st.write('Here we can clearly see that the last campaign was the one which performed better, the marketing team should concentrate on the strategies and channels used in this last campaign or use it as a starting point on which to develop further marketing efforts.')
  ##Correlation betrween data.
  st.subheader('Correlation heatmap')

  fig, ax = plt.subplots(figsize=(15,12))
  sns.heatmap(marketing_data.corr(), annot=True, ax =ax);
  st.write(fig)
  
  st.markdown("""
  **From the correlation heatmap many patterns are enlightened:**

  1. Income:
    - People with higher income tend to spend more and to make more purchases.
    - As Income increase, people tend to purchase more in store or through catalog, while the Web Purchases amount is negatively correlated to the increase in income.
    - In particular, higher income means higher expenditure on products like WIne and Meat Products.

  2. Year of Birth:
    - Year birth tends to be uncorrelated or negatively correlated with all the features involved.
    - As a customers are younger they tend to have lower income, so they purchase less and have less teen or kids at home.

  3. People with kids:
    - spend way less in particular on products like wine.
    - Tend to purchase more using deals such as coupon or discounts.

  4. As the average amount increase:
    - people spend more on products such wines, meat and fish.
    - people tend to purchase throug shops or catalog instead of the website of the company.

  """)

  #Year_Birth_Categories

  def dividing_year_birth(year):
    if year >= 1940 and year <= 1954:
      return '1940-1954'
    elif year > 1954 and year <= 1968:
      return '1955-1968'
    elif year > 1968 and year <= 1982:
      return '1969-1982'
    else:
      return '1983-1996'

  marketing_data['Year_birth_category'] = marketing_data['Year_Birth'].apply(dividing_year_birth)

  #marketing_data[['Year_birth_category', 'Year_Birth']]

with data_visualization:
  
  st.markdown('''
  ## Data Visualization.
  ###### Show interesting plots about the dataset.

  ##### First, we will see some plots to show personal features of the customers:
''')

  fig,ax=plt.subplots(nrows=3,ncols=3,figsize=(15,12))
  sns.countplot(ax=ax[0, 0],x='Education',data=marketing_data)
  sns.countplot(ax=ax[0, 1],x='Marital_Status',data=marketing_data)
  sns.countplot(ax=ax[0, 2],x='Kidhome',data=marketing_data)
  sns.countplot(ax=ax[1, 0],x='Teenhome',data=marketing_data)
  sns.countplot(ax=ax[1,1],x='Year_birth_category', data=marketing_data)
  sns.histplot(ax=ax[1,2], x='Year_Birth', data = marketing_data)
  sns.histplot(ax=ax[2,0], x='Income', data = marketing_data)
  sns.histplot(ax=ax[2,1], x='Tot_amount', data = marketing_data)
  sns.histplot(ax=ax[2,2], x='Tot_purchases', data = marketing_data)

  fig.tight_layout()
  st.pyplot(fig)

  st.subheader('Correlation between amount of wine and average amount purchased')

  wine_income_df = marketing_data[marketing_data['MntWines']!=0][['MntWines', 'Income']]
  meat_income_df = marketing_data[marketing_data['MntMeatProducts']!=0][['MntMeatProducts', 'Income']]

  fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(15,10))
  sns.scatterplot(ax=ax[0],x='MntWines', y='Income', data=wine_income_df)
  sns.scatterplot(ax=ax[1],x='MntMeatProducts',y = 'Income', data=meat_income_df);
  plt.title('Correlation between Income and amount spent on Wine and Meat', x=-0.1, fontsize=15)
  fig.tight_layout()
  st.pyplot(fig)

  st.write('Here it is clear how STRONG is the correlation between the amount of wine and meat purchased and the income of each customer. Being them, especially wine, not a basic necessity, the more a person can spend, the more they are spending on whims such wine and meat. It can be clearly seen here where after a certain point the amount spent on the specific product increase considerably.')

  st.subheader('Marketing campaign performance across different age-categories.')
  Cmp1 = marketing_data[marketing_data['AcceptedCmp1']==1].groupby(['Year_birth_category'])['ID'].nunique()
  Cmp2 = marketing_data[marketing_data['AcceptedCmp2']==1].groupby(['Year_birth_category'])['ID'].nunique()
  Cmp3 = marketing_data[marketing_data['AcceptedCmp3']==1].groupby(['Year_birth_category'])['ID'].nunique()
  Cmp4 = marketing_data[marketing_data['AcceptedCmp4']==1].groupby(['Year_birth_category'])['ID'].nunique()
  Cmp5 = marketing_data[marketing_data['AcceptedCmp5']==1].groupby(['Year_birth_category'])['ID'].nunique()
  Response = marketing_data[marketing_data['Response']==1].groupby(['Year_birth_category'])['ID'].nunique()

  accepted_campaigns = pd.concat([Cmp1, Cmp2, Cmp3, Cmp4, Cmp5, Response], axis=1)
  accepted_campaigns.columns = ['Cmp1', 'Cmp2', 'Cmp3', 'Cmp4', 'Cmp5', 'Response']

  accepted_campaigns

  ax = accepted_campaigns.plot(kind='bar', figsize=(14, 10))
  ax.set_xticks([0, 1, 2, 3])
  ax.set_xticklabels(['1940-1954', '1955-1968', '1969-1982', '1983-1996'], rotation = 0)
  plt.suptitle('Different campaigns effectiveness across Year of Birth categories', fontsize=15, y=0.9);
  
  image2 = Image.open('age.png')
  st.image(image2)

with modeling:
  st.markdown('''
  ## Customer segmentation using K-means clustering

  **The purpose is to divide the data space or data points into a number of groups, such that data points in the same groups are more similar to other data points in the same group, and dissimilar to the data points in other groups.
  Clustering identifies what people do most of the times in order to predict what customers are more likely to do moving forward.**
  ''')
  backup_df = marketing_data.reset_index()

  backup_df.head()

  backup_df = backup_df.iloc[:,2:]

  backup_df['Average amount'] = backup_df['Average amount'].astype(int)

  backup_df

  """We are going to drop the columns 'Dt_Customer' and 'Recency' because are not of interest in our analysis:"""

  backup_df.drop(['Dt_Customer', 'Recency'], axis=1, inplace=True)

  backup_df.info()

  st.write('In order to perform the K-means clustering algorithm, we only need numerical variables given that it performs the clustering of data calculating and trying to minimize the euclidean distance between the k centroids of the clusters and the data points around each centroid. Then it finds the mean of each new cluster to establish a new centroid.')

  numeric_df = backup_df.select_dtypes(include='int')
  st.dataframe(numeric_df)

  #we drop ID column because does not tell anything about the data.
  numeric_df = numeric_df.drop('ID', axis=1)

  st.markdown('''
  #### Optimal number of clusters using the Elbow method:
  In order to define K (i.e. the optimal number of clusters), it is recommended to use the so-called "elbow criterion". It finds the number of clusters for which adding an extra cluster would not add sufficient information or would cause overfitting.
  The elbow plot presents the ratio of within cluster to between clusters on the y-axis and the number of clusters on the x-axis. There will come a point in which the ratio will not decrease so much by adding a new cluster. The elbow plot is the ideal number of clusters.
  ''')
  
  #Elbow method to find the optimal number of k-clusters

  square_distances = []
  x = numeric_df
  for i in range(1,12):
      km = KMeans(n_clusters=i, random_state=42)
      km.fit(x)
      square_distances.append(km.inertia_)

  fig, ax = plt.subplots()
  plt.figure(figsize=(14,10))
  plt.plot(range(1,12), square_distances, 'bx-')
  plt.xlabel('K')
  plt.ylabel('inertia')
  plt.title('Elbow Method')
  plt.xticks(list(range(1,12)))
  plt.show()

  image3 = Image.open('elbow.png')
  st.image(image3)
  

  st.write('Here 3 seems the optimal number of clusters given that after it the difference on the y-axis is slight.')

  #Now we will feat the K-means clustering model on our data 'numeric_df' with 3 clusters. 

  km = KMeans(n_clusters=3, random_state=42)
  backup_df['cluster'] = km.fit_predict(x)
  #after we add the cluster column to the backup_df which is the df with all the numeric and categorical data we had before choosing only the numeric.

  #replacing of 0,1,2 with 1,2,3 so we have Cluster 1, Cluster 2, Cluster 3
  backup_df['cluster'] = backup_df['cluster'].replace({0:1, 1:2, 2:3})

  #Now by putting the cluster into the dataframe it is possible to group them and see the average quantities for each cluster:
  st.write('**Mean values for each cluster:**')
  st.dataframe(backup_df.groupby('cluster').mean().T)

  #Now it is better to take a closer look and analyze each cluster individually and in the end draw conclusions.
  st.text("")
  st.subheader('Clusters in depth')
  st.markdown('''
  
  ##### Cluster 1 in depth:''')

  Cluster_1 = backup_df[backup_df['cluster']==1]
  
  #mean values for cluster 1
  Cluster_1.describe().loc['mean',:]
  st.dataframe(Cluster_1.describe().loc['mean',:])

  fig,ax=plt.subplots(nrows=3,ncols=2,figsize=(12,10))
  sns.countplot(ax=ax[0, 0],x='Education',data=Cluster_1)
  sns.countplot(ax=ax[0, 1],x='Marital_Status',data=Cluster_1)
  sns.countplot(ax=ax[1, 0],x='Kidhome',data=Cluster_1)
  sns.countplot(ax=ax[1, 1],x='Teenhome',data=Cluster_1)
  sns.countplot(ax=ax[2,0], x='Country', data=Cluster_1)
  sns.countplot(ax=ax[2,1], x='Year_birth_category', data=Cluster_1)
  
  fig.tight_layout()
  st.pyplot(fig)

  #Spending Habits
  cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
  for x in cols:
      print('The average {} for cluster 1 is '.format(x), Cluster_1[x].mean())

  st.write('Mean amounts spent by Cluster 1:')
  col1, col2, col3, col4, col5, col6 = st.columns(6)
  col1.metric("Wine", "30.77")
  col2.metric("Fruits", "6.01")
  col3.metric("Meat", "25.71")
  col4.metric("Fish", "9.10")
  col5.metric("Sweets", "6.07")
  col6.metric("Gold", "17.82")

  #Purchasing Behavior
  parameter = ['NumStorePurchases', 'NumWebPurchases', 'NumCatalogPurchases']
  for x in parameter:
      print('The average {} for cluster 1 are '.format(x), Cluster_1[x].mean())
      
  st.write('Purchasing behavior by Cluster 1:')
  col1, col2, col3= st.columns(3)
  col1.metric("Num Store Purchases", "3.1")
  col2.metric("Num Web Purchases", "2.17")
  col3.metric("Num Catalog Purchases", "0.53")
  st.text("")
  st.markdown('''
  
  ##### Cluster 2 in depth:''')

  Cluster_2 = backup_df[backup_df['cluster']==2]
  
  st.dataframe(Cluster_2.describe().loc['mean',:])

  fig,ax=plt.subplots(nrows=3,ncols=2,figsize=(12,10))
  sns.countplot(ax=ax[0, 0],x='Education',data=Cluster_2)
  sns.countplot(ax=ax[0, 1],x='Marital_Status',data=Cluster_2)
  sns.countplot(ax=ax[1, 0],x='Kidhome',data=Cluster_2)
  sns.countplot(ax=ax[1, 1],x='Teenhome',data=Cluster_2)
  sns.countplot(ax=ax[2,0], x='Country', data=Cluster_2)
  sns.countplot(ax=ax[2,1], x='Year_birth_category', data=Cluster_2)

  fig.tight_layout()
  st.pyplot(fig)

  #Spending Habits
  cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
  for x in cols:
      print('The average {} for cluster 2 is '.format(x), Cluster_2[x].mean())
  
  st.write('Mean amounts spent by Cluster 2:')
  col1, col2, col3, col4, col5, col6 = st.columns(6)
  col1.metric("Wine", "618.50")
  col2.metric("Fruits", "57.03")
  col3.metric("Meat", "397.16")
  col4.metric("Fish", "82.60")
  col5.metric("Sweets", "60.12")
  col6.metric("Gold", "70.29")

  #Purchasing Behavior
  parameter = ['NumStorePurchases', 'NumWebPurchases', 'NumCatalogPurchases']
  for x in parameter:
      print('The average {} for cluster 1 are '.format(x), Cluster_2[x].mean())

  st.write('Purchasing behavior by Cluster 2:')
  col1, col2, col3= st.columns(3)
  col1.metric("Num Store Purchases", "8.42")
  col2.metric("Num Web Purchases", "5.43")
  col3.metric("Num Catalog Purchases", "5.44")
  st.text("")
  st.markdown('''
  
  ##### Cluster 3 in depth:''')

  Cluster_3 = backup_df[backup_df['cluster']==3]

  Cluster_3.describe().loc['mean',:]

  fig,ax=plt.subplots(nrows=3,ncols=2,figsize=(12,10))
  sns.countplot(ax=ax[0, 0],x='Education',data=Cluster_3)
  sns.countplot(ax=ax[0, 1],x='Marital_Status',data=Cluster_3)
  sns.countplot(ax=ax[1, 0],x='Kidhome',data=Cluster_3)
  sns.countplot(ax=ax[1, 1],x='Teenhome',data=Cluster_3)
  sns.countplot(ax=ax[2,0], x='Country', data=Cluster_3)
  sns.countplot(ax=ax[2,1], x='Year_birth_category', data=Cluster_3)

  fig.tight_layout()
  st.pyplot(fig)

  #Spending Habits
  cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
  for x in cols:
      print('The average {} for cluster 3 is '.format(x), Cluster_3[x].mean())
  
  st.write('Mean amounts spent by Cluster 3:')
  col1, col2, col3, col4, col5, col6 = st.columns(6)
  col1.metric("Wine", "290.23")
  col2.metric("Fruits", "18.56")
  col3.metric("Meat", " 98.25")
  col4.metric("Fish", "25.13")
  col5.metric("Sweets", "17.80")
  col6.metric("Gold", "45.63")

  #Purchasing Behavior
  parameter = ['NumStorePurchases', 'NumWebPurchases', 'NumCatalogPurchases']
  for x in parameter:
      print('The average {} for cluster 1 are '.format(x), Cluster_3[x].mean())

  st.write('Purchasing behavior by Cluster 2:')
  col1, col2, col3= st.columns(3)
  col1.metric("Num Store Purchases", "6.09")
  col2.metric("Num Web Purchases", "4.74")
  col3.metric("Num Catalog Purchases", "2.26")
  st.text("")
  st.markdown('''
  
  ### After the in depth analysis of clusters we describe our findings to create Buyer Personas'

  #### Cluster 1: Lower Income, Occasionals and Parents of kids.
  - Younger people
  - Low Income
  - Very probable the presence of at least 1 kid.
  - Very low wine amounts.
  - lower expenditure in general.
  - need to catch them with offers and discounts
  - more probable that they will complain, CS need to treat them more careful.
  - They buy mostly in Store and through web in equal measure.

  #### Cluster 2: High Income - Loyals.
  - Highest income
  - Also the more LOYAL with on average 19 purchases per person in the last 2 years.
  - The most of them have NO kids, if they have son they are teenagers.
  - They spend hugely on things like Wine and Meat and on average they spend more also on the other goods.
  - need to offer them some premium services or loyalty prizes in order to keep the customer relationship strong.
  - They buy largely in Store, on average 8 times, but also in discrete quantity via Web or catalog.

  #### Cluster 3: Middle Income, average loyalty and Parents of teens.
  - Middle income
  - Good but not great loyalty with 13 purchases in the last 2 years.
  - Strong presence of teenagers sons in the house.
  - Spend a discrete quantity in wine and groceries but a modest quantity in MeatProducts.
  - The segment who uses the most Deals to purchase.
  - Preferred channel to purchase is via web, on average 5 times.
  ''')

  st.subheader('Marketing campaigns performance across the clusters')
  
  #Grouping for Campaign accepted with number of people for each cluster.
  Camp1 = backup_df[backup_df['AcceptedCmp1']==1].groupby(['cluster'])['ID'].nunique()
  Camp2 = backup_df[backup_df['AcceptedCmp2']==1].groupby(['cluster'])['ID'].nunique()
  Camp3 = backup_df[backup_df['AcceptedCmp3']==1].groupby(['cluster'])['ID'].nunique()
  Camp4 = backup_df[backup_df['AcceptedCmp4']==1].groupby(['cluster'])['ID'].nunique()
  Camp5 = backup_df[backup_df['AcceptedCmp5']==1].groupby(['cluster'])['ID'].nunique()
  CampResponse = backup_df[backup_df['Response']==1].groupby(['cluster'])['ID'].nunique()

  #creation of the dataframe by concatenating the responses created before.
  campaign_performance_cluster = pd.concat([Camp1, Camp2, Camp3, Camp4, Camp5, CampResponse], axis=1)
  campaign_performance_cluster.columns = ['Camp1', 'Camp2', 'Camp3', 'Camp4', 'Camp5', 'CampResponse']

  #Fill null values with 0.
  campaign_performance_cluster.fillna(0, inplace=True)

  st.dataframe(campaign_performance_cluster)

  ax = campaign_performance_cluster.plot(kind='bar', figsize=(14, 10))
  ax.set_xticks([0, 1, 2])
  ax.set_xticklabels(['Cluster 1', 'Cluster2', 'Cluster3'], rotation = 0)
  plt.suptitle('Different campaigns effectiveness across different clusters', fontsize=15, y=0.9);

  image4 = Image.open('cluster_campaign.png')
  st.image(image4)

  st.markdown("""
  The marketing campaigns were very effective (except the second one) on the second segment, especially the last one, the fifth and the first one, this highlights that these campaign were well designed for premium customers and they should use them as a starting point for future ones.

  Largely across all clusters the **last** campaign was the most effective. One fact that stands out is that the 5th campaign were accepted only by the High Income and Loyal cluster, so this highlights once again the success of the campaign in this segment.

  Campaign 3 was more effective in the Lower income cluster, so it is better if the marketing department targets this cluster to develop strategies incentrated to this campaign. Or, for example offering some special offers on cheap products for kids, being this cluster highly concentrated of them.

  In general, people with less kids were more responsive to the campaigns so, if the company prefers to specialize on these type of customers they are performing really well. But, if they want to make people spend more on their products even if they have less disposable income they should start to diversificate the offering of products with more cheap products or special offers well targeted to each segment.
  """)

