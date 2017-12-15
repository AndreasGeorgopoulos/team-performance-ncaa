#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andreas Georgopoulos
"""

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup 
from itertools import product
import sklearn.metrics
import seaborn as sns
import pandas as pd
import numpy as np
import httplib2
import html5lib
import pylab

# Import Data
bball_data = pd.read_csv("college_bball_data.csv")

# Unique Colleges that we have data for
colleges = list(set(bball_data['School']))

""" 
-------------  Get coaches data of each college from 2006-2015-----------------
"""

http = httplib2.Http()
coaches_wl_per_college = {}

for year in range(min(bball_data['Year'])-1, max(bball_data['Year'])+1):
    # Create url of corresponding year to acquire coach data
    url = 'http://www.sports-reference.com/cbb/seasons/{}-coaches.html'.format(year)
    # Get HTML
    status, response = http.request(url)
    soup = BeautifulSoup(response, "html5lib")
    
    # Parse HTML file and extract coach data for each college
    coach_data_wl = {}
    for row in soup.find_all('tr'):
        try:
            # 
            if row.contents[1].get_text().lower() in colleges: 
                coach_data_wl[row.contents[1].get_text().lower()] = {}
                coach_data_wl[row.contents[1].get_text().lower()]['Name'] = row.contents[0].get_text() 
                coach_data_wl[row.contents[1].get_text().lower()]['First_Coaching_Year'] = row.contents[9].get_text() 
                coach_data_wl[row.contents[1].get_text().lower()]['Overall_Win_Loss_Percentage'] = row.contents[12].get_text()    # Win - Loss Percentage (=Total Wins/ Total Losses, if >0/5 more wins)
                coach_data_wl[row.contents[1].get_text().lower()]['NCAA_appearances']    = row.contents[19].get_text()            # Number of NCAA Tournament appearances
        except:
            pass
    coaches_wl_per_college[year] = coach_data_wl

""" 
    Get sequential coaching years  in corresponding team of each coach per college till 'current' year
        & W-L% of coache's carrer overall till 'current' year
"""
coach_years_wl = pd.DataFrame(list(product(list(set(bball_data['School'])), list(set(bball_data['Year'])))), columns=['School', 'Year']) 
coach_years_wl['Coach_Name'] = 'NA'
coach_years_wl['Coaching_Years'] = 0
coach_years_wl['Coach_Win_Loss_Perc'] = 0
coach_years_wl['Coach_NCAA_appear'] = 0
              
for i in range(len(coach_years_wl)):
    school = coach_years_wl.loc[i,'School']
    year   = coach_years_wl.loc[i,'Year']
    # Find coach of team/college in year
    coach_years_wl.loc[i,'Coach_Name'] = coaches_wl_per_college[year][school]['Name']            
    # Compute sequential coaching years of coach in corresponding team/college until 'current' year and append to df
    coach_years_wl.loc[i,'Coaching_Years'] = int(year) - int(coaches_wl_per_college[year][school]['First_Coaching_Year']) +1    
    # Find Win - Loss % of coach based on his overall carreer till 'current' year (take W-L% prev year)
    coach_years_wl.loc[i,'Coach_Win_Loss_Perc'] = float(coaches_wl_per_college[year-1][school]['Overall_Win_Loss_Percentage'])    
    # Find coach NCAA appearances till 'current' year
    if coaches_wl_per_college[int(year)-1][school]['NCAA_appearances'] == '': 
        coach_years_wl.loc[i,'Coach_NCAA_appear'] = 0 
    else: 
        coach_years_wl.loc[i,'Coach_NCAA_appear'] = int(coaches_wl_per_college[int(year)-1][school]['NCAA_appearances'])
      

""" 
--------------- Get team composition of each college per year -----------------
          years: 2005-2015 (since need comparisons from previous year)
"""
team_composition = {}
for college in colleges:
    team_comp_year = {}
    for year in range(min(bball_data['Year'])-1, max(bball_data['Year'])+1):
    
        # Create url of corresponding college to acquire coach data
        url = 'http://www.sports-reference.com/cbb/schools/{}/{}.html'.format(college,year)
        # Get HTML
        status, response = http.request(url)
        soup = BeautifulSoup(response, "html5lib")
        # Parse HTML file and extract team composition for corresponding year and college
        team = []
        for row in soup.find_all('th'):
            try:
                team.append(row.contents[0].get_text())
            except:
                pass
        # dictionary of each year's team composition for college
        team_comp_year[year] =team
    # dictionary of each college's team composition per year
    team_composition[college] = team_comp_year
                    
               
""" 
    Find Jaccard similarity of each college's team composition from previous year
    (capture the change in team's composition from year to year)
"""

def similarity(a, b):
    """
    Returns the simlarity of two lists as the number of elements in the 
    intersection of the two lists divided by the number of elements in the 
    union of the two lists
    """
    return len(list(set(a).intersection(b))) / len(list(set(a).union(b))) if len(list(set(a).union(b))) else 0
    
# Dataframe of all combinations of school and year where the team composition change ariable will be computed for   
similarity_team_composition = pd.DataFrame(list(product(list(set(bball_data['School'])), list(set(bball_data['Year'])))), columns=['School', 'Year']) 
similarity_team_composition['Team_Comp_Change'] = 0
                           
for i in range(len(similarity_team_composition)):
    school = similarity_team_composition.loc[i,'School']
    year   = similarity_team_composition.loc[i,'Year']
    
    # extract team composition for current year and previous year of the school
    team_comp_cur_year  = team_composition[school][year]
    team_comp_prev_year = team_composition[school][(year-1)]
    # find similarity between team compositions and append to corresponding year
    similarity_team_composition.loc[i,'Team_Comp_Change'] = round(similarity(team_comp_cur_year, team_comp_prev_year),4)

# Save additional variables df
coach_years_wl.to_csv('coach_years_wl.csv')
similarity_team_composition.to_csv('similarity_team_composition.csv')    



"""
    Join Dataframe with Additional Variables -----------------------------------
"""
# Read csv of additiona variables
#coach_years_wl = pd.read_csv('coach_years_wl.csv').iloc[:,1:7]
#similarity_team_composition = pd.read_csv('similarity_team_composition.csv').iloc[:,1:4]


bball_data = pd.merge(bball_data, coach_years_wl, on = ['School','Year'], how = 'left')
bball_data = pd.merge(bball_data, similarity_team_composition, on = ['School','Year'], how = 'left')
#bball_data['Coach_NCAA_appear'] = bball_data['Coach_NCAA_appear'].where((pd.notnull(bball_data['Coach_NCAA_appear'])), 0)


"""
    Data Exploration -----------------------------------------------------------
"""
# Find columns with NA values:
bball_data.columns[bball_data.isnull().any()].tolist()  

# Replace spaces,':" with '_' in column names
bball_data.columns = [x.strip().replace(' ', '_') for x in bball_data.columns]
bball_data.columns = [x.strip().replace(':', '_') for x in bball_data.columns]

# Target Variable
target = 'outcome'
# Predictors
predictors = list(bball_data.columns)
predictors = [e for e in predictors if e not in {'School','opp_name','Percentage_Dif_Points','Coach_Name','pts','opp_pts','outcome'}]


"""
    Check distribution of each independent variable (identify problems of skewness etc)
"""
bball_data_tidy = (
    bball_data[predictors].stack() # pull the columns into row variables   
      .to_frame()               
      .reset_index()               # pull the resulting MultiIndex into the columns
      .rename(columns={0: 'Value'})  # rename the unnamed column
)
bball_data_tidy.columns.values[1] = 'Var:'

sns.set(style="white", color_codes=True)
g = sns.FacetGrid(bball_data_tidy, col='Var:', col_wrap=5, sharey=False, sharex=False, 
                  margin_titles=True )
g = g.map(plt.hist, "Value", alpha=.4)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Distribution of Predictor Variables', fontsize=14)
g.savefig("hist_multipleplot.png")


"""
    Check relationship of each independent variable to the dependent
"""
bball_data_tidy.columns.values[0] = 'Index_Merge'
bball_data_target = bball_data[['Game','outcome']]
bball_data_target['Index_Merge'] = list(bball_data_target.index.values)
                              
bball_data_tidy_new = pd.merge(bball_data_tidy, bball_data_target, on = 'Index_Merge', how = 'left')
                              
sns.set(style="white", color_codes=True)
g2 = sns.FacetGrid(bball_data_tidy_new, col='Var:', col_wrap=5, sharey=False, sharex=False, 
                  margin_titles=True )
g2 = g2.map(plt.scatter, "outcome", "Value")
g2.fig.subplots_adjust(top=0.9)
g2.fig.suptitle('Predictors on Target Variable', fontsize=14)
g2.savefig("rel_multipleplot.png")


"""
    Check for multicollinearity (calculate VIF) --------------------------------
"""

#VIF is calculated by auxiliary regression, so not dependent on the actual fit. IMHO, diagnostics like this are a good motivation for breaking out the design matrix.
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor, reset_ramsey

features = "+".join(predictors)
y, X = dmatrices("outcome ~" +  features, data= bball_data, return_type="dataframe")

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

# Ommit correlated var based on AIC (R script)
predictors = [e for e in predictors if e not in {'School','opp_name','Percentage_Dif_Points','Coach_Name','pts','opp_pts','outcome','same_conf','experience_in_arena','conference_size_squared'}]
features = "+".join(predictors)


""" 
    Correlation  Heatmap Seaborn
"""
sns.set(style="white")
corr = bball_data.corr()
f2=plt.figure(figsize=(12,8))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    sns.heatmap(corr, mask=mask, vmax=.3, square=True, cmap="RdBu_r")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Correlation Heatmap of all Features")
plt.savefig('Correlation Heatmap.png', bbox_inches='tight')
plt.show()  


"""
    Pairwise Scatterplot Correlation Matrix
"""
# Scatterplot Matrix
g = sns.pairplot(bball_data)
g.fig.suptitle('Pairwise Scatterplot Matrix of Predictors', fontsize=14)
g.savefig("scatterplot_matrix.png")


"""
    Random Forest  ------------------------------------------------------------
"""
## Random Forest
def random_forest_func(df, predictors, targets, test_size=.4):

    #Split into training and testing sets
    x = df[predictors]
    y = df[targets]
    pred_train, pred_test, tar_train, tar_test  = train_test_split(x, y, test_size=.4)
    
    #Build model on training data  
    classifier=RandomForestClassifier(n_estimators=25)
    classifier=classifier.fit(pred_train,tar_train)
    
    # fit an Extra Trees model to the data
    model = ExtraTreesClassifier()
    model.fit(pred_train,tar_train)
    
    #getting the training and test accuracies from random forest classifier
    #train_accuracy
    predictions = classifier.predict(pred_train)
    train_accuracy = sklearn.metrics.accuracy_score(tar_train, predictions)
    #test_accuracy
    predictions = classifier.predict(pred_test)
    test_accuracy = sklearn.metrics.accuracy_score(tar_test, predictions)
    
    # display the relative importance of each attribute from extra trees classifier
    feature_importance = model.feature_importances_

    return feature_importance, train_accuracy, test_accuracy


# Random Forest Find importance
importance,train_accuracy,test_accuracy = random_forest_func(bball_data,predictors,target) 
 
# Barplot of importance
def plot_importance(rank, regressors,chart_title):
    # Sort predictors by importance level
    var_imp = rank.tolist()
    sorted_predictors = pd.DataFrame(np.matrix([var_imp,regressors]).T)
    sorted_predictors[0] = sorted_predictors[0].astype(float)
    sorted_predictors = sorted_predictors.sort_values(by = 0, ascending = False)

    # Barplot of importance
    fig, ax = plt.subplots()

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    pos = pylab.arange(len(sorted_predictors))+.5     
    plt.barh(pos,sorted_predictors[0], align='center')
    plt.yticks(pos, sorted_predictors[1])
    plt.xlabel('Variable Importance')
    plt.title(chart_title)
    plt.savefig(chart_title, bbox_inches='tight')
    plt.show()

plot_importance(importance, predictors, "Random Forest - Extra Tree Classifier Results")


"""
    Additional Feature Selection Techinques ------------------------------------------
"""
from sklearn.feature_selection import f_regression
from sklearn.linear_model import RandomizedLasso
from sklearn.ensemble import RandomForestRegressor

    
X = bball_data[predictors].as_matrix()
Y = np.array(bball_data['outcome'])
names =  bball_data[predictors].columns.values

"""
    Random Forest Regressor
"""
rf = RandomForestRegressor()
rf.fit(X,Y)
plot_importance(rf.feature_importances_, predictors,"Random Forest Regressor Results")

"""
    F regression
"""
f_test, var_p_values = f_regression(X , Y, center=True)
plot_importance(f_test, predictors,"F - Regression Results")

"""
    Randomised Lasso - Feature Selection (Stability)
"""
rlasso = RandomizedLasso(alpha=0.025)
rlasso.fit(X, Y)
plot_importance(np.abs(rlasso.scores_), predictors, "Randomised Lasso Results")




"""
    Final Feature Selection ---------------------------------------------------
"""
bball_data['conference_size_squared'] = bball_data['conference_size']**2
bball_data['Coaching_Years_squared'] = bball_data['Coaching_Years']**2
bball_data['class_diversity_squared'] = bball_data['class_diversity']**2
          
predictors = list(bball_data.columns)
features_final = 'home_game + height_diversity + log(ethnic_diversity)  + class_diversity + class_diversity_squared + conference_size + conference_size_squared + conf_wl + opp_games_played + opponent_win_loss + Coach_Win_Loss_Perc + Coach_NCAA_appear + Team_Comp_Change + Coaching_Years + Coaching_Years_squared'
predictors_final = [e for e in predictors if e not in {'School','opp_name','Percentage_Dif_Points','Coach_Name','pts','opp_pts','outcome','same_conf','experience_in_arena','Year','Game','region_diversity','school_diversity','degree_diversity','experience_with_opponent'}]

                                                
"""
    Functional Form Misspecification-------------------------------------------
    
    Ramsey's RESET specification test for linear models: general specification
    test, for additional non-linear effects in a model
    
"""
import statsmodels.formula.api as smf

# Fit regression model
model_final = smf.ols("outcome ~" +  features_final, data=bball_data).fit() # Skip Warnings of multicollinearity--> reflect the square of conference size, no problem
print(model_final.summary())

reset_ramsey(model_final, 3) # fail to reject at 5% sign ---> so no misspecification


"""
    Breusch Bagan test to check for Heteroscedasticity ------------------------
"""
from statsmodels.compat import lzip
import statsmodels.stats.api as sms

name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
test = sms.het_breushpagan(model_final.resid, model_final.model.exog)
lzip(name, test)           # Heterokedastic error ---> go for WLS regression

    
"""
    Final model Robust Errors
"""
# Fit regression model
model_final_robust = smf.ols("outcome ~" +  features_final, data=bball_data).fit(cov_type = 'HC1') # Skip Warnings of multicollinearity--> reflect the square of conference size, no problem
print(model_final_robust.summary())


"""
    Plot percentage contribution -----------------------------------------------
"""
# Percentage contribution of each variables (R_script)
perc_contr = pd.read_csv('perc_contr.csv')

def plot_contribution(rank, regressors,chart_title):
    # Sort predictors by importance level
    var_imp = rank.tolist()
    sorted_predictors = pd.DataFrame(np.matrix([var_imp,regressors]).T)
    sorted_predictors[0] = sorted_predictors[0].astype(float)
    sorted_predictors = sorted_predictors.sort_values(by = 0, ascending = False)

    # Barplot of importance
    fig, ax = plt.subplots()
    
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    pos = pylab.arange(len(sorted_predictors))+.5     
    plt.barh(pos,sorted_predictors[0], align='center')
    plt.yticks(pos, sorted_predictors[1])
    plt.xlabel('% of R-squared')
    plt.ylabel('R-squared =32.58%, metrics are normalised to sum 100%')
    plt.title(chart_title)
    plt.savefig(chart_title, bbox_inches='tight')
    plt.show()

plot_contribution(np.array(perc_contr['Value']), np.array(perc_contr['Variable']), 'Relative importances for Outcome')


"""
    Plot Regression Results ----------------------------------------------------
"""
# give input model.final, predictors_final, list of variables that have log form

log_variables = ['ethnic_diversity']
square_variables = ['conference_size', 'Coaching_Years', 'class_diversity']
# remove squared var from predictors
predictors_final = [e for e in predictors_final if e not in {'conference_size_squared'}]


# Estimate outcome per predictor (rest predictors constant, mean value)
outcome_estimation = pd.DataFrame(columns = predictors_final)
for var in predictors_final: # Target variable: var

    # For Rest Independent variables: compute mean value and plug in with estimated regression coefficients
    sumprod_estimated_mean = 0
    for pred in predictors_final:
        if pred != var:
            # if predictor in log form
            if pred in log_variables:
                sumprod_estimated_mean += np.mean(np.log(bball_data[pred])) * model_final.params['log({})'.format(pred)]
            else:
                sumprod_estimated_mean += np.mean(bball_data[pred]) * model_final.params[pred]

    if var in log_variables:
        outcome_estimation[var] =  np.log(bball_data[var]) * model_final.params['log({})'.format(var)] + sumprod_estimated_mean
    elif var in square_variables:
        outcome_estimation[var] =  bball_data[var] * model_final.params[var] + bball_data['{}_squared'.format(var)] * model_final.params['{}_squared'.format(var)] +  + sumprod_estimated_mean
    else:
        outcome_estimation[var] =  bball_data[var] * model_final.params[var] + sumprod_estimated_mean
    

# Plot
for pred in predictors_final:
    sns.plt.scatter( bball_data[pred],outcome_estimation[pred])

# Stack dataframes                              
outcome_estimation_tidy = (outcome_estimation.stack().to_frame().reset_index().rename(columns={0: 'outcome'}))
outcome_estimation_tidy.columns.values[1] = 'Var:'
                                                            
bball_data_estim_tidy = (bball_data[predictors_final].stack().to_frame().reset_index().rename(columns={0: 'Variable'}))
bball_data_estim_tidy.columns.values[1] = 'Var:'

# Merge dataframes in order to plot Facetgrid in Seaborn
outcome_estimation_tidy_new = pd.merge(bball_data_estim_tidy, outcome_estimation_tidy, on = ['level_0','Var:'], how = 'left')

# Plot                         
sns.set(style="white", color_codes=True)
g3 = sns.FacetGrid(outcome_estimation_tidy_new, col='Var:', col_wrap=4, sharey=False, sharex=False, margin_titles=True )
g3 = g3.map(plt.scatter,"Variable", "outcome")

g3.fig.subplots_adjust(top=0.9)
g3.fig.suptitle('Estimated Regression Results', fontsize=14)

g3.savefig("regres_result_plot.png")
                               

