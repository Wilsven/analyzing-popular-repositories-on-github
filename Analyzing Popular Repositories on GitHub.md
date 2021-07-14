# Analyzing Popular Repositories on GitHub

## Introduction

GitHub is arguably one of the most popular version control and source code management platforms out there. It is also home to one of the largest social networking sites for programmers and a platform where software developers can showcase their skills and past project works to recruiters and hiring managers.

In this report, we will be analyzing data on GitHub repositories to try to gain invaluable insights and information like user behaviour. We will also aim to answer questions like what contributes to a repository's popularity, what technologies and topics are trending among developers nowadays, and much more.

*The [‘GitHub Repositories 2020’](https://www.kaggle.com/vatsalparsaniya/github-repositories-analysis) dataset from Kaggle was used in this project as it is the most up-to-date dataset.*

### Objectives

1. Read, clean and structure data to make it suitable for analysis
2. Analyze top repositories based on popularity
3. Analyze users with many repositories
4. Understand contribution activities across the repositories
5. Explore most popular tags

## Data Exploration & Implementation

Import necessary libraries and reading the input data, 


```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# Read input data
github_data_df = pd.read_csv("Github_data.csv")

# Dropping duplicated columns
github_data_df = github_data_df.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)

# Display basic information about the dataset
github_data_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1500 entries, 0 to 1499
    Data columns (total 19 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   topic             1500 non-null   object 
     1   name              1500 non-null   object 
     2   user              1500 non-null   object 
     3   star              1500 non-null   object 
     4   fork              1500 non-null   object 
     5   watch             1500 non-null   object 
     6   issue             1500 non-null   object 
     7   pull_requests     1500 non-null   object 
     8   projects          1313 non-null   float64
     9   topic_tag         1500 non-null   object 
     10  discription_text  1480 non-null   object 
     11  discription_url   669 non-null    object 
     12  commits           1500 non-null   object 
     13  branches          1500 non-null   object 
     14  packages          1500 non-null   int64  
     15  releases          1500 non-null   int64  
     16  contributers      1500 non-null   object 
     17  License           1500 non-null   object 
     18  url               1500 non-null   object 
    dtypes: float64(1), int64(2), object(16)
    memory usage: 222.8+ KB


*As you can see, the dataframe contains 19 columns, out of which, 11 columns were deemed as important contributors to the popularity of GitHub repositories, as well as its relevance* 

The column "contributers" is a typo so let's correct it and rename the columns for better clarity. 


```python
# Get relevant columns out
github_df = github_data_df[["topic", "name", "user", "star", "fork", "watch", 
                                 "issue", "pull_requests", "topic_tag", "commits", "contributers"]]

# Pass dictionary with oldnames as keys and newnames as values to rename function
newnames = ["Topic", "Repo_Name", "User_Name", "Star", "Fork", "Watch", 
            "Issues", "Pull_Requests", "Topic_Tags", "Commits", "Contributors"]
oldnames = github_df.columns
github_df = github_df.rename(columns = dict(zip(oldnames, newnames)))

# Take a look at first five rows 
github_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic</th>
      <th>Repo_Name</th>
      <th>User_Name</th>
      <th>Star</th>
      <th>Fork</th>
      <th>Watch</th>
      <th>Issues</th>
      <th>Pull_Requests</th>
      <th>Topic_Tags</th>
      <th>Commits</th>
      <th>Contributors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Data-Science</td>
      <td>keras</td>
      <td>keras-team</td>
      <td>47.9k</td>
      <td>18.1k</td>
      <td>2.1k</td>
      <td>2,940</td>
      <td>87</td>
      <td>['deep-learning', 'tensorflow', 'neural-networ...</td>
      <td>5,342</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Data-Science</td>
      <td>scikit-learn</td>
      <td>scikit-learn</td>
      <td>40.3k</td>
      <td>19.6k</td>
      <td>2.2k</td>
      <td>1,505</td>
      <td>764</td>
      <td>['machine-learning', 'python', 'statistics', '...</td>
      <td>25,405</td>
      <td>108</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Data-Science</td>
      <td>PythonDataScienceHandbook</td>
      <td>jakevdp</td>
      <td>23.1k</td>
      <td>9.9k</td>
      <td>1.5k</td>
      <td>65</td>
      <td>67</td>
      <td>['scikit-learn', 'numpy', 'python', 'jupyter-n...</td>
      <td>234</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Data-Science</td>
      <td>Probabilistic-Programming-and-Bayesian-Methods...</td>
      <td>CamDavidsonPilon</td>
      <td>21k</td>
      <td>6.6k</td>
      <td>1.4k</td>
      <td>127</td>
      <td>38</td>
      <td>['bayesian-methods', 'pymc', 'mathematical-ana...</td>
      <td>1,079</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Data-Science</td>
      <td>Data-Science--Cheat-Sheet</td>
      <td>abhat222</td>
      <td>18.4k</td>
      <td>8.2k</td>
      <td>1.5k</td>
      <td>7</td>
      <td>0</td>
      <td>[]</td>
      <td>363</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### A quick brief about the columns in the data:

* **Topic** – A label describing the field or domain of the repo
* **Repo_Name** – Name of the repo (short name for repository)
* **User_Name** – Repository owner’s name
* **Star** – Number of stars a repo has received
* **Fork** – Number of times a repo has been forked
* **Watch** – Number of users watching the repo
* **Issues** – Number of open issues
* **Pull_Requests** – Total pull requests raised
* **Topic_Tags** – List of topic tags added to that repo by user
* **Commits** – Total number of commits made
* **Contributors** – Number of people contributing to the repo

*Notice how in the **Star, Fork** and **Watch** columns, the values are in thousands. Let's clean the data to remove the 'k's, show the full number and convert to 'float' datatype.*

*Also notice how the **Issues** and **Commits** columns have ',' in their entries. Let's also remove those.*


```python
# Converting Star, Fork and Watch columns to numeric by replacing 'k's with 1000
github_df["Star"] = github_df["Star"].apply(lambda x: float(x.rstrip("k"))*1000 if "k" in str(x) else float(x))
github_df["Fork"] = github_df["Fork"].apply(lambda x: float(x.rstrip("k"))*1000 if "k" in str(x) else float(x))
github_df["Watch"] = github_df["Watch"].apply(lambda x: float(x.rstrip("k"))*1000 if "k" in str(x) else float(x))

# Remove ',' from Issues and Commits columns
github_df["Issues"] = github_df["Issues"].apply(lambda x: x.replace(",", ""))
github_df["Commits"] = github_df["Commits"].apply(lambda x: x.replace(",", ""))

# Convert Issues, Pull_Requests, Commits, Contributors columns from object to numeric
cols = ["Issues", "Pull_Requests", "Commits", "Contributors"]
github_df[cols] = github_df[cols].apply(pd.to_numeric, errors="coerce", axis=1)

# Check
github_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1500 entries, 0 to 1499
    Data columns (total 11 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Topic          1500 non-null   object 
     1   Repo_Name      1500 non-null   object 
     2   User_Name      1500 non-null   object 
     3   Star           1500 non-null   float64
     4   Fork           1500 non-null   float64
     5   Watch          1500 non-null   float64
     6   Issues         1499 non-null   float64
     7   Pull_Requests  1499 non-null   float64
     8   Topic_Tags     1500 non-null   object 
     9   Commits        1500 non-null   float64
     10  Contributors   1495 non-null   float64
    dtypes: float64(7), object(4)
    memory usage: 129.0+ KB


Now that the necessary columns are numeric, we can pass them through the *describe()* function to obtain statistical information.


```python
# Display basic statistical details about the columns
github_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Star</th>
      <th>Fork</th>
      <th>Watch</th>
      <th>Issues</th>
      <th>Pull_Requests</th>
      <th>Commits</th>
      <th>Contributors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1499.000000</td>
      <td>1499.000000</td>
      <td>1500.000000</td>
      <td>1495.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8058.556667</td>
      <td>1845.994667</td>
      <td>380.881333</td>
      <td>116.228152</td>
      <td>15.527685</td>
      <td>2535.753333</td>
      <td>28.657525</td>
    </tr>
    <tr>
      <th>std</th>
      <td>17031.342041</td>
      <td>4572.215953</td>
      <td>754.014969</td>
      <td>344.511792</td>
      <td>55.227818</td>
      <td>21605.630660</td>
      <td>79.740051</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>576.000000</td>
      <td>109.000000</td>
      <td>37.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>50.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1800.000000</td>
      <td>385.500000</td>
      <td>108.000000</td>
      <td>14.000000</td>
      <td>2.000000</td>
      <td>196.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9025.000000</td>
      <td>1700.000000</td>
      <td>416.000000</td>
      <td>72.500000</td>
      <td>9.000000</td>
      <td>953.500000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>310000.000000</td>
      <td>80900.000000</td>
      <td>8400.000000</td>
      <td>4337.000000</td>
      <td>851.000000</td>
      <td>797646.000000</td>
      <td>877.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 1. Analysis of Top Repositories Based on Popularity

We will try to use the **Star, Fork** and **Watch** columns to answer why a GitHub repository is popular.

#### Recap:

* Star – When a user likes the repo or they want to show some appreciation, they Star it.
* Watch – When a user wants to be notified of all the activities in a repo, then they Watch it.
* Fork – When a user wants a copy of the repo or they intend to make any contributions, they Fork it.


```python
# Create a dataframe with average values of the columns across all topics
pop_mean_df = github_df.groupby("Topic").mean().reset_index()

# Display dataframe
pop_mean_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic</th>
      <th>Star</th>
      <th>Fork</th>
      <th>Watch</th>
      <th>Issues</th>
      <th>Pull_Requests</th>
      <th>Commits</th>
      <th>Contributors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Android-studio</td>
      <td>121.43</td>
      <td>39.42</td>
      <td>10.07</td>
      <td>7.120000</td>
      <td>0.350000</td>
      <td>113.66</td>
      <td>3.900000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Arduino</td>
      <td>1979.69</td>
      <td>839.55</td>
      <td>168.55</td>
      <td>82.390000</td>
      <td>12.210000</td>
      <td>1083.87</td>
      <td>21.410000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Computer-Vision</td>
      <td>4925.00</td>
      <td>1537.83</td>
      <td>283.24</td>
      <td>133.220000</td>
      <td>14.150000</td>
      <td>1462.67</td>
      <td>10.270000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Data-Science</td>
      <td>6669.00</td>
      <td>2145.86</td>
      <td>400.22</td>
      <td>146.950000</td>
      <td>20.620000</td>
      <td>2732.74</td>
      <td>29.561224</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GAN</td>
      <td>2440.45</td>
      <td>668.75</td>
      <td>122.66</td>
      <td>41.120000</td>
      <td>3.230000</td>
      <td>425.29</td>
      <td>9.850000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Open-CV</td>
      <td>271.49</td>
      <td>133.17</td>
      <td>26.17</td>
      <td>5.210000</td>
      <td>2.150000</td>
      <td>40.56</td>
      <td>0.820000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Raspberry pi</td>
      <td>1515.87</td>
      <td>330.78</td>
      <td>124.25</td>
      <td>47.520000</td>
      <td>4.750000</td>
      <td>9041.52</td>
      <td>29.410000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>awesome</td>
      <td>1653.00</td>
      <td>279.10</td>
      <td>113.20</td>
      <td>14.200000</td>
      <td>2.090000</td>
      <td>205.36</td>
      <td>0.940000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>c++</td>
      <td>19380.35</td>
      <td>3656.02</td>
      <td>874.50</td>
      <td>234.800000</td>
      <td>30.310000</td>
      <td>6349.22</td>
      <td>52.595960</td>
    </tr>
    <tr>
      <th>9</th>
      <td>flutter</td>
      <td>3556.83</td>
      <td>666.05</td>
      <td>131.05</td>
      <td>53.373737</td>
      <td>4.920000</td>
      <td>638.29</td>
      <td>20.240000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>java</td>
      <td>20430.00</td>
      <td>5726.86</td>
      <td>1116.52</td>
      <td>225.700000</td>
      <td>32.810000</td>
      <td>4373.26</td>
      <td>57.750000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>javascript</td>
      <td>41961.00</td>
      <td>7044.71</td>
      <td>1421.73</td>
      <td>438.510000</td>
      <td>61.920000</td>
      <td>6717.30</td>
      <td>152.428571</td>
    </tr>
    <tr>
      <th>12</th>
      <td>machine-Learning</td>
      <td>15186.00</td>
      <td>4390.00</td>
      <td>853.75</td>
      <td>283.000000</td>
      <td>40.666667</td>
      <td>4088.71</td>
      <td>31.980000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>sensor</td>
      <td>733.37</td>
      <td>217.57</td>
      <td>62.66</td>
      <td>28.740000</td>
      <td>2.870000</td>
      <td>742.90</td>
      <td>11.350000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>variational-encoder</td>
      <td>54.87</td>
      <td>14.25</td>
      <td>4.65</td>
      <td>0.940000</td>
      <td>0.120000</td>
      <td>20.95</td>
      <td>0.090000</td>
    </tr>
  </tbody>
</table>
</div>



We can see that some of the top repository topics are **JavaScript, Java, C++ etc.**

### Analysis of Star column

Let's visualize the average number of stars across topics.


```python
# Let's sort the Star column in the dataframe for a nicer barplot
sorted_star_df = pop_mean_df.sort_values("Star", ascending=False)

# Set figure size and dpi
fig, ax = plt.subplots(figsize=(6,4), dpi=100)

# Add color to edge
plt.rcParams["axes.edgecolor"]="#333F4B"

# Customize spines and tick parameters
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="both", which="both", labelsize=10, bottom=True, left=False)
ax.set_xlim(0,45000)

# Remove grid and set background color
ax.grid(False)
ax.set_facecolor("white")

# Plot the data and set palette
sns.barplot(data=sorted_star_df, x="Star", y="Topic", palette=("Blues_d"))

# Set x and y-axis labels and title
ax.set_xlabel("No. of Stars", fontsize=13, color = "#333F4B")
ax.set_ylabel("Topic", fontsize=13, color = "#333F4B")
fig.suptitle("Average Stars on Each Topic", fontsize=18, color = "#333F4B")
plt.show()
```


    
![png](output_18_0.png)
    


Now let's take a look at the top 10 most starred repositories.


```python
# Top 10 most starred repos
github_df.nlargest(n=10, columns="Star")[["Repo_Name", "Topic", "Star"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Repo_Name</th>
      <th>Topic</th>
      <th>Star</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1000</th>
      <td>freeCodeCamp</td>
      <td>javascript</td>
      <td>310000.0</td>
    </tr>
    <tr>
      <th>1001</th>
      <td>vue</td>
      <td>javascript</td>
      <td>163000.0</td>
    </tr>
    <tr>
      <th>1002</th>
      <td>react</td>
      <td>javascript</td>
      <td>147000.0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>tensorflow</td>
      <td>machine-Learning</td>
      <td>144000.0</td>
    </tr>
    <tr>
      <th>1003</th>
      <td>bootstrap</td>
      <td>javascript</td>
      <td>140000.0</td>
    </tr>
    <tr>
      <th>1004</th>
      <td>You-Dont-Know-JS</td>
      <td>javascript</td>
      <td>120000.0</td>
    </tr>
    <tr>
      <th>1100</th>
      <td>coding-interview-university</td>
      <td>c++</td>
      <td>110000.0</td>
    </tr>
    <tr>
      <th>800</th>
      <td>CS-Notes</td>
      <td>java</td>
      <td>98400.0</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>CS-Notes</td>
      <td>c++</td>
      <td>98400.0</td>
    </tr>
    <tr>
      <th>801</th>
      <td>javascript</td>
      <td>java</td>
      <td>94800.0</td>
    </tr>
  </tbody>
</table>
</div>



The most starred repository is **FreeCodeCamp** in the topic **JavaScript** with **310000.0 stars**.

### Analysis of Watch column

Let's visualize the average number of watchers across topics.


```python
# Let's sort the Watch column in the dataframe for a nicer barplot
sorted_watch_df = pop_mean_df.sort_values("Watch", ascending=False)

# Set figure size and dpi
fig, ax = plt.subplots(figsize=(6,4), dpi=100)

# Add color to edge
plt.rcParams["axes.edgecolor"]="#333F4B"

# Customize spines and tick parameters
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="both", which="both", labelsize=10, bottom=True, left=False)
ax.set_xlim(0,1600)

# Remove grid and set background color
ax.grid(False)
ax.set_facecolor("white")

# Plot the data
sns.barplot(data=sorted_watch_df, x="Watch", y="Topic", palette=("Blues_d"))

# Set x and y-axis labels and title
ax.set_xlabel("No. of Watchers", fontsize=13, color = "#333F4B")
ax.set_ylabel("Topic", fontsize=13, color = "#333F4B")
fig.suptitle("Average Watchers on Each Topic", fontsize=18, color = "#333F4B")
plt.show()
```


    
![png](output_23_0.png)
    



```python
# Top 10 repos with the most watchers
github_df.nlargest(n=10, columns="Watch")[["Repo_Name", "Topic", "Watch"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Repo_Name</th>
      <th>Topic</th>
      <th>Watch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>tensorflow</td>
      <td>machine-Learning</td>
      <td>8400.0</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>freeCodeCamp</td>
      <td>javascript</td>
      <td>8400.0</td>
    </tr>
    <tr>
      <th>1003</th>
      <td>bootstrap</td>
      <td>javascript</td>
      <td>7200.0</td>
    </tr>
    <tr>
      <th>1133</th>
      <td>CodeHub</td>
      <td>c++</td>
      <td>6800.0</td>
    </tr>
    <tr>
      <th>1002</th>
      <td>react</td>
      <td>javascript</td>
      <td>6700.0</td>
    </tr>
    <tr>
      <th>1001</th>
      <td>vue</td>
      <td>javascript</td>
      <td>6100.0</td>
    </tr>
    <tr>
      <th>1100</th>
      <td>coding-interview-university</td>
      <td>c++</td>
      <td>5800.0</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>free-programming-books-zh_CN</td>
      <td>javascript</td>
      <td>5500.0</td>
    </tr>
    <tr>
      <th>1004</th>
      <td>You-Dont-Know-JS</td>
      <td>javascript</td>
      <td>5400.0</td>
    </tr>
    <tr>
      <th>800</th>
      <td>CS-Notes</td>
      <td>java</td>
      <td>4800.0</td>
    </tr>
  </tbody>
</table>
</div>



The repository with the most watchers is **TensorFlow** in the topic **Machine Learning** with **8400 watchers**.

### Analysis of Fork column

Let's visualize the average number of forks across topics.


```python
# Let's sort the Watch column in the dataframe for a nicer barplot
sorted_fork_df = pop_mean_df.sort_values("Fork", ascending=False)

# Set figure size and dpi
fig, ax = plt.subplots(figsize=(6,4), dpi=100)

# Add color to edge
plt.rcParams["axes.edgecolor"]="#333F4B"

# Customize spines and tick parameters
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="both", which="both", labelsize=10, bottom=True, left=False)
ax.set_xlim(0,8000)

# Remove grid and set background color
ax.grid(False)
ax.set_facecolor("white")

# Plot the data
sns.barplot(data=sorted_fork_df, x="Fork", y="Topic", palette=("Blues_d"))

# Set x and y-axis labels and title
ax.set_xlabel("No. of Forks", fontsize=13, color = "#333F4B")
ax.set_ylabel("Topic", fontsize=13, color = "#333F4B")
fig.suptitle("Average Forks on Each Topic", fontsize=18, color = "#333F4B")
plt.show()
```


    
![png](output_27_0.png)
    



```python
# Top 10 repos with the most forks
github_df.nlargest(n=10, columns="Fork")[["Repo_Name", "Topic", "Fork"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Repo_Name</th>
      <th>Topic</th>
      <th>Fork</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>tensorflow</td>
      <td>machine-Learning</td>
      <td>80900.0</td>
    </tr>
    <tr>
      <th>1003</th>
      <td>bootstrap</td>
      <td>javascript</td>
      <td>68700.0</td>
    </tr>
    <tr>
      <th>300</th>
      <td>opencv</td>
      <td>Computer-Vision</td>
      <td>34900.0</td>
    </tr>
    <tr>
      <th>800</th>
      <td>CS-Notes</td>
      <td>java</td>
      <td>32100.0</td>
    </tr>
    <tr>
      <th>1100</th>
      <td>coding-interview-university</td>
      <td>c++</td>
      <td>32100.0</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>CS-Notes</td>
      <td>c++</td>
      <td>32100.0</td>
    </tr>
    <tr>
      <th>806</th>
      <td>spring-boot</td>
      <td>java</td>
      <td>29600.0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>courses</td>
      <td>Data-Science</td>
      <td>28500.0</td>
    </tr>
    <tr>
      <th>1002</th>
      <td>react</td>
      <td>javascript</td>
      <td>28500.0</td>
    </tr>
    <tr>
      <th>802</th>
      <td>JavaGuide</td>
      <td>java</td>
      <td>26300.0</td>
    </tr>
  </tbody>
</table>
</div>



The repository with the most forks is **TensorFlow** in the topic **Machine Learning** with **80900.0 forks**.

### Relationship Between Star, Watch and Fork columns

When a user has interest in a repository, there is a possibility where they will also give it a star, watch or fork it. But how much do these factors relate to one another? Let's explore that next.

### Star and Fork


```python
# Set figure size and dpi
fig, ax = plt.subplots(figsize=(8,4), dpi=100)

# Set seaborn theme for background grids
sns.set_theme("paper")

# Plot the data
sns.regplot(data=github_df, x="Star", y="Fork", color="#4682B4")

# Set x and y-axis labels and title
ax.set_xlabel("No. of Stars", fontsize=13, color="#333F4B")
ax.set_ylabel("No. of Forks", fontsize=13, color="#333F4B")
fig.suptitle("Relationship Between Star and Fork", fontsize=18, color="#333F4B")
plt.show()
```


    
![png](output_32_0.png)
    


### Star and Watch


```python
# Set figure size and dpi
fig, ax = plt.subplots(figsize=(8,4), dpi=100)

# Set seaborn theme for background grids
sns.set_theme("paper")

# Plot the data
sns.regplot(data=github_df, x="Star", y="Watch", color="#4682B4")

# Set x and y-axis labels and title
ax.set_xlabel("No. of Stars", fontsize=13, color="#333F4B")
ax.set_ylabel("No. of Watchers", fontsize=13, color="#333F4B")
fig.suptitle("Relationship Between Star and Watch", fontsize=18, color="#333F4B")
plt.show()
```


    
![png](output_34_0.png)
    


### Fork and Watch


```python
# Set figure size and dpi
fig, ax = plt.subplots(figsize=(8,4), dpi=100)

# Set seaborn theme for background grids
sns.set_theme("paper")

# Plot the data
sns.regplot(data=github_df, x="Watch", y="Fork", color="#4682B4")

# Set x and y-axis labels and title
ax.set_xlabel("No. of Watchers", fontsize=13, color="#333F4B")
ax.set_ylabel("No. of Forks", fontsize=13, color="#333F4B")
fig.suptitle("Relationship Between Watch and Fork", fontsize=18, color="#333F4B")
plt.show()
```


    
![png](output_36_0.png)
    


After plotting the relationships between the Star, Watch and Fork columns, we can see that the data points fit much closer to the regression line in the Fork and Watch plot as compared to the other two plots.

**From this, we can conclude that if a user is watching a repo, he or she is more likely to fork it as well.**

## 2. Analysis of Users with Large Number of Repositories

Let's take a look at users who have a large number of repositories.


```python
# Create a list of top 10 users with large number of repositories
large_repo_users = github_df.groupby("User_Name").size().nlargest(n=10).reset_index(name="Count")["User_Name"].to_list()

# Create a dataframe using users_large_repos list
large_repo_users_df = github_df[github_df["User_Name"].isin(large_repo_users)][["Topic", "User_Name", "Star"]]

# Set figure size and dpi
fig, ax = plt.subplots(figsize=(8,4), dpi=100)

# Add color to edge
plt.rcParams["axes.edgecolor"]="#333F4B"

# Customize spines and tick parameters
sns.despine(left=True, bottom=True)
ax.tick_params(axis="both", which="both", labelsize=10, bottom=False, left=False)
ax.set_xlim(0, 18)

# Remove grid and set background color
ax.grid(False)
ax.set_facecolor("white")

# Plot the data
sns.countplot(data=large_repo_users_df, y="User_Name", 
              order=large_repo_users_df["User_Name"].value_counts().index, palette=("Blues_d"))

# Set x and y-axis labels and title
ax.set_xlabel("Count", fontsize=13, color = "#333F4B")
ax.set_ylabel("User", fontsize=13, color = "#333F4B")
fig.suptitle("Top 10 Users with Most Number of Repositories", fontsize=18, color = "#333F4B")
plt.show()
```


    
![png](output_39_0.png)
    


From the plot above, we can tell that:

* *Microsoft tops the list with 17 repos.*
* *Followed by Google with 15 repos.*
* *Of all the users in the top 10, three seem to be individual users (junyanz, rasbt, MicrocontrollersAndMore)*

## 3. Understanding Contribution Activities across the Repositories

We can achieve this by plotting a heatmap to study the correlation between features in the data.


```python
# Check if there are null values
github_df.isnull().sum()
```




    Topic            0
    Repo_Name        0
    User_Name        0
    Star             0
    Fork             0
    Watch            0
    Issues           1
    Pull_Requests    1
    Topic_Tags       0
    Commits          0
    Contributors     5
    dtype: int64




```python
# Drop rows with any null values and create a dataframe with only the contribution columns
corr_df = github_df.dropna(axis=0, subset=["Issues", "Pull_Requests", "Contributors"])[["Issues", "Pull_Requests", "Commits", "Contributors"]]

# Set figure size and dpi
fig, ax = plt.subplots(figsize=(6,4), dpi=100)

# Customize tick parameters
ax.tick_params(labelsize=13, labelrotation=90)


# Plot the correlation in a heatmap
sns.heatmap(corr_df.corr(), linewidths=0.1, vmax=1.0, square=True, linecolor="white", annot=True, cmap="Blues")
fig.suptitle("Correlation Between the Contribution Columns", fontsize=16, color = "#333F4B")
plt.show()
```


    
![png](output_43_0.png)
    


From the heatmap above, we can tell that none of the parameters are strongly correlated. In fact, most of them are not dependent on any parameters. Perhaps the only two parameters with moderate positive correlation are the number of issues and pull requests.

Now, let us explore the correlation in the top 100 most popular repositories to study if there are no correlation as well.


```python
popular_df = github_df.nlargest(n=100, columns="Star")[["Issues", "Pull_Requests", "Commits", "Contributors"]]

# Set figure size and dpi
fig, ax = plt.subplots(figsize=(6,4), dpi=100)

# Customize tick parameters
ax.tick_params(labelsize=13, labelrotation=90)


# Plot the correlation in a heatmap
sns.heatmap(popular_df.corr(), linewidths=0.1, vmax=1.0, square=True, linecolor="white", annot=True, cmap="Blues")
fig.suptitle("Correlation of Contribution Columns in Top 100 Popular Repositories",fontsize=14, color = "#333F4B")
plt.show()
```


    
![png](output_45_0.png)
    


The results for the top 100 repositories does not differ much from the previous heatmap plot comprising of the entire dataset. So let's move on to users with large number of repositories instead and examine if there are any correlations between the parameters. 


```python
# Create a dataframe using large_repo_users list
large_repo_users_corrdf = github_df[github_df["User_Name"].isin(large_repo_users)][["Issues", "Pull_Requests", "Commits", "Contributors"]]

# Set figure size and dpi
fig, ax = plt.subplots(figsize=(6,4), dpi=100)

# Customize tick parameters
ax.tick_params(labelsize=13, labelrotation=90)


# Plot the correlation in a heatmap
sns.heatmap(large_repo_users_corrdf.corr(), linewidths=0.1, vmax=1.0, square=True, linecolor="white", annot=True, cmap="Blues")
fig.suptitle("Correlation of Contribution Columns of Users with Large No. of Repositories",fontsize=14, color = "#333F4B")
plt.show()
```


    
![png](output_47_0.png)
    


Unsurprisingly, users who are more active tend to have large number of repositories. From the heatmap above, we can see that there is a strong correlation between:

* Commits and pull requests
* Commits and issues
* Pull requests and issues

**Therefore, conclusions we can make are that there are no real relationships between contributon activities in the overall dataset, no correlation among contributon activities in the top 100 most popular repositories and users with many repositories tend to be contribute more.**

## 4. Analysis of Topic Tags

Adding topic tags to a repository helps with classification. It helps other users to easily find and contribute to relevant repos which are properly tagged. It also helps you and users nagivate through the platform much easily, looking for specifc repos by topic, type and interest. 

The Topic_Tags column consists of lists. To find the number of popular tags, we can first group the data into a list of lists and count the occurrence of each tag. Then, with this data, we can perform visualization processes to examine the popularity of the tags.


```python
# Take a look at first five entries of Topic_Tags column
github_df[["Topic_Tags"]].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic_Tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>['deep-learning', 'tensorflow', 'neural-networ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>['machine-learning', 'python', 'statistics', '...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>['scikit-learn', 'numpy', 'python', 'jupyter-n...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>['bayesian-methods', 'pymc', 'mathematical-ana...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[]</td>
    </tr>
  </tbody>
</table>
</div>




```python
import ast
from collections import Counter

# Convert column to list of lists
topic_tag_list = github_df["Topic_Tags"].apply(lambda x: ast.literal_eval(x)).to_list()

# Split all lists and select individual items
all_tags = [tag for item in topic_tag_list for tag in item]

# Count the occurrence of each tag
tag_dict = Counter(all_tags)

# Create a dataframe with top 15 most tags
toptags_df = pd.DataFrame(tag_dict.most_common(15), columns=["Name of the Tag", "Count"])

# View first five rows of dataframe
toptags_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name of the Tag</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>machine-learning</td>
      <td>236</td>
    </tr>
    <tr>
      <th>1</th>
      <td>deep-learning</td>
      <td>214</td>
    </tr>
    <tr>
      <th>2</th>
      <td>python</td>
      <td>149</td>
    </tr>
    <tr>
      <th>3</th>
      <td>computer-vision</td>
      <td>127</td>
    </tr>
    <tr>
      <th>4</th>
      <td>javascript</td>
      <td>123</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Set figure size and dpi
fig, ax = plt.subplots(figsize=(7,4), dpi=100)

# Rotate the labels along x-axis
plt.xticks(rotation=90)
ax.grid(False)

# Customize spine, remove grid and set background color
sns.despine()
ax.grid(False)
ax.set_facecolor("white")

# Plot the data
sns.barplot(data=toptags_df, x="Name of the Tag", y="Count", palette=("Blues_d"))

# set x and y-axis labels and title
ax.set_xlabel("Topic Tags", fontsize=13, color = "#333F4B")
ax.set_ylabel("Count", fontsize=13, color = "#333F4B")
fig.suptitle("Top 15 Most Popular Topic Tags", fontsize=18, color = "#333F4B")
plt.show()
```


    
![png](output_52_0.png)
    



```python
# Length of tags list in each column
len_tags = [len(tag) for tag in topic_tag_list]

# Create a new column
github_df["Total_Tags"] = len_tags

# Group based on topic and calculate total_tags in each topic
topic_wise_tag = github_df.groupby("Topic").sum()["Total_Tags"].reset_index(name="Total Tags")

# View first five rows of total tags dataframe grouped by topics
topic_wise_tag.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic</th>
      <th>Total Tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Android-studio</td>
      <td>482</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Arduino</td>
      <td>577</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Computer-Vision</td>
      <td>935</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Data-Science</td>
      <td>911</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GAN</td>
      <td>618</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Sort the dataframe in descending order to display nicer barplot 
topic_wise_tag = topic_wise_tag.sort_values(ascending=False, by="Total Tags").reset_index().drop(["index"], axis=1)

# Set figure size and dpi
fig, ax = plt.subplots(figsize=(7,4), dpi=100)

# Rotate the labels along x-axis

# Remove background grids
ax.grid(False)
ax.set_facecolor('white')
sns.despine()

# Plot the data
sns.barplot(data=topic_wise_tag, x="Total Tags", y="Topic", ci=None, palette=("Blues_d"))

# set x and y-axis labels and title
ax.set_xlabel("Total Tags", fontsize=13, color = "#333F4B")
ax.set_ylabel("Topic", fontsize=13, color = "#333F4B")
fig.suptitle("Tags Distribution Across Topics", fontsize=18, color = "#333F4B")
plt.show()
```


    
![png](output_54_0.png)
    


**From the bar plot above, we can see that repositories with topics Computer Vision, Data Science, and Machine Learning tend to be tagged more.**

Now, let's finish up with a wordcloud of topic tags!


```python
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

# Create a string with all the topic tags
github_tags = (" ").join(all_tags)

# Instantiate a word cloud object
tags_wc = WordCloud(
    background_color="white",
    max_font_size=120,
    max_words=200)

# Generate the word cloud
tags_wc.generate(github_tags)

# Plot wordcloud and set title
plt.figure(figsize=(6,6),dpi=100)
plt.imshow(tags_wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.title("Most common tags used in Github Repositories",fontdict={"size": 15, "color": "#333F4B", "verticalalignment": "center"})
plt.show()
```


    
![png](output_56_0.png)
    


## Conclusion

If we had data going back a decade ago, I am certain the trend would look quite different. From this report, we can see that in recent years, topics involving Data Science has become increasingly popular. For instance, interests in topics like Machine Learning, Deep Learning, Python and Computer Vision have seen their popularity grown in numbers over the past few years! 


```python

```
