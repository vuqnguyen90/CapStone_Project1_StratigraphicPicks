
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import numpy as np

from sklearn import svm
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.dummy import DummyClassifier

from numpy import mean
from numpy import std
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score


from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek


import plotly.express as px
import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.graph_objs import *
init_notebook_mode()

from sklearn.model_selection import learning_curve
from imblearn.pipeline import Pipeline

no_split=10
no_repeat=3


# Processs raw dataframes. Including change the column names, convert to the right data type
# Merge 2 dataframes together
# Define Features and Target matrix
def process_data(intellog,picks):
    # Rename 'Pick' column to Depth to merge with intellog dataframe by merging SitID and Depth
    picks.columns=['SitID','HorID','Depth','Quality']
    picks['Depth']=pd.to_numeric(picks['Depth'], errors='coerce')
    intellog['RW']=pd.to_numeric(intellog['RW'], errors='coerce')
    main_file=pd.merge(intellog,picks,how='inner',on=['SitID','Depth'])
    main_file=main_file.dropna()
    
    features=main_file.drop(['SitID','HorID','Depth','LithID'],axis=1)
    target=main_file['LithID']
  
   
    return main_file,features, target


def get_sampling_technique():
    
    sampling_technique=list()
    sampling_name=list()
    
    # RandomOverSampler 
    sampling_technique.append( RandomOverSampler(random_state=123))
    sampling_name.append('RandomOverSampler')
    
    # SMOTE
    sampling_technique.append( SMOTE(random_state=123))
    sampling_name.append('SMOTE')
    
    # ADASYN 
    sampling_technique.append( ADASYN(random_state=123))
    sampling_name.append('ADASYN')
    
    # Downsampling tech
    sampling_technique.append( RandomUnderSampler(random_state=123))
    sampling_name.append('RandomUnderSampler')
    
    #SMOTEENN
    sampling_technique.append(SMOTEENN(random_state=123))
    sampling_name.append('SMOTEENN')
    
    #SMOTETomek
    sampling_technique.append(SMOTETomek(random_state=123))
    sampling_name.append('SMOTETomek')


    # Combine Over and Undersampling Methods
    over=RandomOverSampler(random_state=123)
    under=RandomUnderSampler(random_state=123)
    
    sampling_technique.append(Pipeline(steps=[('o',over),('u',under)]))
    sampling_name.append('Over-Under Resampling Combination')
    
    
    # Combine Over and Undersampling Methods
    smote=SMOTE(random_state=123)
    under=RandomUnderSampler(random_state=123)
    
    sampling_technique.append(Pipeline(steps=[('smote',over),('u',under)]))
    sampling_name.append('SMOTE-Under Resampling Combination')
    
    return sampling_technique, sampling_name
    

def target_distribution_visualization(data,target,name): 
    
    ncolor=7
    # Create an array with the colors you want to use
    colors = ['#A68FBF', '#FFFF00','#008744','#E1812E','#070C57','#0A67AD','#F7347A']
    # Set your custom color palette
    customPalette=sns.set_palette(sns.color_palette('RdBu',n_colors=ncolor))
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sns.scatterplot(data.iloc[:,1],data.iloc[:,0],hue=target,palette=sns.color_palette(colors,7))
    plt.title('Classes Distribution with ' + str(name))
    
       
  
# To evaluate the model performance with different resampling techniques. 
def evaluate_model_sampling_technique(x, y, model,s,resampling_name):

    rkf = RepeatedStratifiedKFold(n_splits=no_split, n_repeats=no_repeat, random_state=123)
    f1score=list()

    for train_idx, val_idx in rkf.split(x,y): 
        x_train,x_val=x.iloc[train_idx],x.iloc[val_idx]
        y_train,y_val=y.iloc[train_idx],y.iloc[val_idx]
        
        
        x_train_res,y_train_res=s.fit_resample(x_train,y_train)
        
        data_res=pd.DataFrame(x_train_res)
        
        
        model.fit(x_train_res,y_train_res)
        
        val_preds=model.predict(x_val.values)
        f1score.append(f1_score(y_val, val_preds, average='macro'))

    target_distribution_visualization(data_res,y_train_res,resampling_name)

    return model,f1score


def evaluate_model_baseline(x, y, model):

    rkf = RepeatedStratifiedKFold(n_splits=no_split, n_repeats=no_repeat, random_state=123)
    score=list()

    for train_idx, val_idx in rkf.split(x,y): 
        x_train,x_val=x.iloc[train_idx],x.iloc[val_idx]
        y_train,y_val=y.iloc[train_idx],y.iloc[val_idx]
        
        
        # model=LogisticRegression(solver='newton-cg',multi_class='multinomial',max_iter=200)
        model.fit(x_train,y_train)
        
        val_preds=model.predict(x_val)
        score.append(f1_score(y_val, val_preds, average='macro'))
        
        
        ave_f1score=np.mean(score)
        
        
    return model,score  


def plot_learning_curve(x,y,model):

    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(model, 
                                                            x, 
                                                            y,
                                                            # Number of folds in cross-validation
                                                            cv=3,
                                                            # Evaluation metric
                                                            scoring='f1_macro',
                                                            # Use all computer cores
                                                            n_jobs=-1, 
                                                            # 50 different sizes of the training set
                                                            train_sizes=np.linspace(0.01, 1.0, 50))
    
    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Draw lines
    plt.figure()
    # plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
    
    # Draw bands
    # plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
    
    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("F1_Macro"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    
    
    
######################################### Main Codes#########################################
intellog=pd.read_csv('INTELLOG.csv')
picks=pd.read_csv('PICKS.csv')

data, features,target=process_data(intellog,picks)

# Count the values in the target
# to see the imbalanced class distribution
counter=Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%d, Count=%d, Percentage= %.2f%%' % (k, v, per))
    
# to split dataset into training and test dataset. Note that the distribution of class labels 
# are the same by using 'stratify' option

scale=StandardScaler()
scale.fit(features)
features_scale=scale.transform(features)


features_scale=pd.DataFrame(features_scale)


features_scale.columns=features.columns

x_train,x_test,y_train,y_test=train_test_split(features_scale,target,test_size=0.3, random_state=8,stratify=target)




######################################################################################################################################
# Base Case Performance 
    
rf = RandomForestClassifier()

m,scores=evaluate_model_baseline(x_train, y_train, rf)

train_score=np.average(scores)

test_pred_baseline=m.predict(x_test)
test_score_bs=(f1_score(test_pred_baseline,y_test,average='macro'))

print(classification_report(test_pred_baseline,y_test))


# Compute Important Features Using Random Forest 
importance=m.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar(x_train.columns, importance)
plt.show()
plt.title('Important Features for Original Dataset')
plt.xlabel('Feature')


## Use baseline Random Forest to train and fit with the 2 most important
## features
x_train_2=x_train[['VSH','PHI']]
x_test_2=x_test[['VSH','PHI']]
rf = RandomForestClassifier()
m2,scores2=evaluate_model_baseline(x_train_2, y_train, rf)

train_score_2=np.average(scores2)

test_pred_baseline_2=m2.predict(x_test_2)
test_score_bs_2=(f1_score(test_pred_baseline_2,y_test,average='macro'))


## Use baseline Random Forest to train and fit with the 3 most important
x_train_3=x_train[['W_Tar','VSH','PHI']]
x_test_3=x_test[['W_Tar','VSH','PHI']]
rf = RandomForestClassifier()
m3,scores3=evaluate_model_baseline(x_train_3, y_train, rf)

train_score_3=np.average(scores3)

test_pred_baseline_3=m3.predict(x_test_3)
test_score_bs_3=(f1_score(test_pred_baseline_3,y_test,average='macro'))


# Defined Two Additional Feature Engineers. Train and Fit baseline Random Forest
# with these features engineers in combination with exisitng features to see if it
# improve the model's performance 

x_train_f=x_train.copy()
x_train_f['VS']=1-x_train_f['VSH']
x_train_f['Shc']=1-x_train_f['SW']

x_test_f=x_test.copy()
x_test_f['VS']=1-x_test_f['VSH']
x_test_f['Shc']=1-x_test_f['SW']

rf = RandomForestClassifier()

mf,scoresf=evaluate_model_baseline(x_train_f, y_train, rf)

train_score_f=np.average(scoresf)

test_pred_baseline_f=mf.predict(x_test_f)
test_score_bs_f=(f1_score(test_pred_baseline_f,y_test,average='macro'))

importance=m.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar(x_train_f.columns, importance)
plt.show()
plt.title('Important Features with Additional Engineer Features')
plt.xlabel('Feature')


# To Fit and train with new engineering features
x_train_f_5=x_train_f[['W_Tar','SW','VSH','PHI','RW','VS']]
x_test_f_5=x_test_f[['W_Tar','SW','VSH','PHI','RW','VS']]

rf = RandomForestClassifier()

mf5,scoresf5=evaluate_model_baseline(x_train_f_5, y_train, rf)

train_score_f_5=np.average(scoresf5)

test_pred_baseline_f_5=mf5.predict(x_test_f_5)
test_score_bs_f_5=(f1_score(test_pred_baseline_f_5,y_test,average='macro'))

name=['Original_Data','2 Important_Features','3 Important_Features','Original+ 2 Engineer_Features','EngineerFeature_5 Important features']
score=[test_score_bs,test_score_bs_2,test_score_bs_3,test_score_bs_f,test_score_bs_f_5]


# Model's performance comparison between different choices features
plt.figure()
plt.bar(name,score)
plt.show()
plt.title('The Performance of Model with different combination of features ')
plt.ylabel('Macro F-1 Score')

######################################################################################################################################

# Applied different sampling techniques with the baseline RF to see which if sampling method improve the technique 

rf = RandomForestClassifier()

sampler,sample_name=get_sampling_technique()


test_score=list()
for idx, s in enumerate(sampler):
    model, scores=evaluate_model_sampling_technique(x_train, y_train, rf,s,sample_name[idx])
    test_pred=model.predict(x_test)
    test_score.append(f1_score(test_pred,y_test,average='macro'))
    
    
test_score.insert(0,test_score_bs)   
names=['No Resample','RandomOverSampler','SMOTE','ADASYN','RandomUnderSampler','SMOTEENN','SMOTETomek','Over-Under','SMOTE-Under' ] 

test_result=pd.DataFrame({'TestScore':test_score, 'Resampling':names})
fig=px.bar(test_result,x='Resampling',y='TestScore')
plot(fig)  
   

#############################################################################################################################################
## Fine Tune RF model with Random and GridSearchCV 

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


# Fine tune hyperameter for Random Forest
# Random Search Grid to find the range for hyperparameters
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 2, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = [int(x) for x in np.linspace(start = 1, stop = 20, num = 1)]
# Maximum number of levels in tree  
max_depth = [int(x) for x in np.linspace(2, 100, num = 2)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 4,6,8, 10,12,14,16,18,20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8,10,12,14]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
'max_features': max_features,
'max_depth': max_depth,
'min_samples_split': min_samples_split,
'min_samples_leaf': min_samples_leaf,
'bootstrap': bootstrap}

print(random_grid)
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring='f1_macro', n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
rf_random.fit(x_train,y_train)


print (rf_random.best_params_)
# {'n_estimators': 556, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 1, 'max_depth': 50, 'bootstrap': True}
print(rf_random.best_score_)

results = pd.DataFrame(rf_random.cv_results_)


## Use Grid Search to fine tune the hyperparameter

# Create the parameter grid based on the results of random search
param_grid = {'n_estimators': [550,556,560,565],
'max_features': [1,2],
'max_depth': [45,50,55],
'min_samples_split': [1,2,3],
'min_samples_leaf': [1,2,3],
'bootstrap': bootstrap
}

   
    # Create a based model
rf = RandomForestClassifier()
    # Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, scoring='f1_macro',cv = 5, n_jobs = -1, verbose = 2)
    
# Fit the grid search to the data
grid_search.fit(x_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

test_pred=grid_search.best_estimator_.predict(x_test)
print(f1_score(y_test,test_pred,average='macro'))



### To Evaluate Random Forest model with fine-tune Hyper-parameters
rf = RandomForestClassifier(max_depth= 50, max_features= 2, min_samples_leaf= 1, min_samples_split= 3, n_estimators= 556, bootstrap=False)
plot_learning_curve(x_train,y_train,rf)

rf.fit(x_train,y_train)
test_pred=rf.predict(x_test)
print(f1_score(y_test,test_pred,average='macro'))


test_pred=rf.predict(x_test)
print(f1_score(test_pred,y_test,average='macro'))

plot_learning_curve(x_train,y_train,rf)


