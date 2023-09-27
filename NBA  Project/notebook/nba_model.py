#Algorithm for selecting team

import random
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from rid_utils import show_distribution

plt.style.use("ggplot")
rid_seed = 12

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


#visualisations
def plot_heatmap(data):
    """
    This shows the visualisation of the correlation between features
    """
    fig, ax = plt.subplots(figsize=(16,10))
    sns.heatmap(
        data=data,
        annot=True,
        linecolor="black",
        linewidths=0.5,
        cmap="YlGnBu",
        ax=ax
    )
    plt.title("Correllation Plot", color='blue')
    plt.show()
    

def plot_dotplot(x, y, data):
    """
    This shows the visualisation of the data points
    """
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(
        x=data[x],
        y=data[y],
        hue=data["Conference"],
        ax=ax
    )
    plt.title(f"Scatter Plot for {x} and {y}", color='brown')
    plt.show()

    

def build_model(data, type_model="tree"):
    """
    used to build model using traditional algorithms. the algorithms can be selected to either be linear or tree
    
    - Arguments:
        data; a pd.DataFrame object or a json in which the visualisation is to be shown
        type_model; these help to select the type of algorithm to be used. Default; "tree"
    - Returns:
        model; .joblib file of the model built.
        mae; the mean squared error of the model.
        y_pred; predictions obtained
    """
    rid_seed = 12
    X = data.drop(columns=["PTS", "DRB", "team", "Alias", "Conference"])
    y = data["win_rate"]

    X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.2, random_state=rid_seed)
    
    #selecting model to be used
    if type_model == "tree":
        rfr = RandomForestRegressor(random_state=rid_seed, max_depth=100)
        bagging = BaggingRegressor(base_estimator=rfr, n_estimators=5, random_state=rid_seed)
        model_to_be_used = bagging
    elif type_model == "linear1":
        ela = ElasticNet(alpha=0.1, max_iter=100, random_state=rid_seed)
        model_to_be_used = ela
    elif type_model == "linear2":
        lin = LinearRegression()
        model_to_be_used = lin
    elif type_model == "linear3":
        rg = Ridge()
        model_to_be_used = rg
    elif type_model == "gb":
        gb =  GradientBoostingRegressor(random_state=rid_seed)
        model_to_be_used = gb
        
        
    model = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        StandardScaler(),
        model_to_be_used
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, mae, y_pred


pathX = "C:/Users/DELL/Project/FreeWorks/NBA  Project/data/training.csv"
pathY = "C:/Users/DELL/Project/FreeWorks/NBA  Project/model.joblib"

def algo_predict(data,
                 check_data = pd.read_csv(pathX,
                                          index_col=None), 
                 team_only_performance = "one",
                 all=False):
    
    """The functions help predict the likelihood of teams to make it to  the play-offs. Depending on the arguement
    the function is capable of predicting in so many ways options available are further described in the arguments specified
    below.
    
    - Arguments:
            data; a pd.DataFrame object or a json in which you would input to get predictions.
            check_data; this the training data in which the model is trained on.
            team_only_performance; This help in selecting the type of operation needed. Default; "one"
            all; to tell which team would be shown
    - Returns:
        The prediction required. Default; False
    """
    
    if team_only_performance == "one":
        #single team prediction
        df = data.copy()
        df = df.drop(columns=["PTS", "DRB", "team", "Alias", "Conference"])
        model = joblib.load(pathY)
        y_pred = model.predict(df)
        print(f"The win rate of the team is {y_pred}")
        
        if y_pred[0] > check_data["win_rate"].mean() * 0.8:
            print("Team might likely make it to the play offs")
            if y_pred >= check_data["win_rate"].max()*0.8:
                print("Team might likely make it to the Semis")
        else:
            print("Team might not make it")
            
    elif team_only_performance == "group":
        
        #single team prediction in groups
        df = data.copy()
        df = df.drop(columns=["PTS", "DRB", "team", "Alias", "Conference"])
        model = joblib.load(pathY)
        y_pred = model.predict(df)
        predictions = pd.DataFrame({"team":data["team"], "conf": data["Conference"], "Predictions": y_pred})
        
        return predictions
         
    elif team_only_performance == "prob":
        #team prediction in groups
        df = data.copy()
        df = df.drop(columns=["PTS", "DRB", "team", "Alias", "Conference"])
        model = joblib.load(pathY)
        y_pred = model.predict(df)
        predictions = pd.DataFrame(
            {"team": data["team"],
             "win_prob": data["win_prob_avg"],
             "points": data["PTS"],
             "win_rate": data["win_rate"],
             "conf": data["Conference"],
             "predictions": y_pred}
        )
        
        eastern = predictions[predictions["conf"] == "E"].sort_values(by="predictions", ascending=False)
        western = predictions[predictions["conf"] == "W"].sort_values(by="predictions", ascending=False)
        
        eastern_play_off = eastern.head(10) 
        western_play_off = western.head(10)
        
        if all:
            return eastern_play_off, western_play_off, eastern, western
        else:
            return eastern, western