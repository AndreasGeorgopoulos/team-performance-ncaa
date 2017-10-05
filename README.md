# Features influencing team performance in NCAA

At this project a dataset from The National Collegiate Athletic Association ([`college_bball_data.csv`](https://github.com/AndreasGeorgopoulos/team-performance-ncaa/blob/master/college_bball_data.csv)) consisting of basketball games for Ivy League colleges from 2006 to 2015 is analysed in order to identify the features that drive a team’s performance. The main code is presented at the [`Python_Code.py`](https://github.com/AndreasGeorgopoulos/team-performance-ncaa/blob/master/Python_Code.py) file and depicts the following procedures:

### Data Acquisition
The initial dataset is supplemented with additional features, which are engineered based on the additional data that is acquired from the generated web-scraping algorithms. The data that is extracted from the [sport-reference](http://www.sports-reference.com/cbb) website can be sumarised as following:  

**1. Coach Data**  
For each year and school the following coach data is acquired:
- Coach Name
- First coaching year at corresponding team
- Number of NCAA appearances
- Win:Loss percentage 

From the aforementioned coach data the following additional features are generated for each of the target teams that are present in the initial dataset:
- Consecutive years coaching corresponding team until corresponding year
- Win:Loss percentage of coach based on his overall carreer until corresponding year (overall W-L% of previous year)
- Total number of coach NCAA appearances until corresponding year
  
**2. Team Composition Data**  
Similarly, for each year and school, the names of correspodning players are extracted in order to capture the capture the change in each team's composition from year to year. More specifically, the Jaccard similarity of each school and year is computed based on the team composition of the corresponding year and the previous one.

### Feature Selection 
All variables are examined for correlation between each other by visualising a pairwise correlation scatterplot matrix as well as by VIF calculations in order to avoid any problem of multicollinearity. When two highly correlated variables are identified, the one that is included at the model with the lowest AIC is preferred and the other is omitted from the set of potential predictors. In order to identify the most important variables a univariate selection method namely F – Regression, a Random Forest Regressor, and a stability Randomised Lasso selection process are implemented. In addition, the functional form of each variable is selected based on the distribution of each one of them and is validated through statistical tests (Ramsey's RESET test) to avoid any bias due to functional form misspecification. 

An additional code file ([`R_code_BFSelection.R`](https://github.com/AndreasGeorgopoulos/team-performance-ncaa/blob/master/R_code_BFSelection.R)) is provided , which depicts a forward-backward feature selection process implemented in R as well as a weighted least square estimation of the selected regressors on the target variable of the outcome of each game (metric of team performance).

### Regression
After selecting the most important features that explain the variation of a team’s performance on each game, a linear regression model with robust standard errors (due to heteroskedastic standard errors) that describes the outcome of a game in terms of the selected predictors is generated.

*** To read full report [click here](http://www.andreasgeorgopoulos.com/team-performance-ncaa/)

