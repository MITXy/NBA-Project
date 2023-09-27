#utility library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def show_category(x, y, data, z="Pos", kind="small"):
    """
    -Produces a large plot of the data
    """
    if kind == "big":
        fig, ax = plt.subplots(figsize=(60,6))
        sns.boxplot(
            x=x,
            y=y,
            data=data,
            hue=z,
            ax=ax
        )
        plt.xlabel(f"{x}")
        plt.ylabel(f"{y}")
        plt.title(f"The {y} by {x} while observing {z}")
        plt.show();
        
    elif kind == "small":
        fig, ax = plt.subplots(figsize=(12,4))
        sns.boxplot(
            x=x,
            y=y,
            data=data.head(30),
            hue=z,
            ax=ax
        )
        plt.xlabel(f"{x}")
        plt.ylabel(f"{y}")
        plt.title(f"The {y} by {x} while observing {z}")
        plt.show();



def wrangle_player_data(path):
    """
    This function wrangles the data and also creates the  Rating column
    
    - Input: Takes in the player data as input
    
    - Output: The cleaned data
    """
    #importing data and some wrangling
    data = pd.read_csv(path, index_col=[1])
    data.drop("Unnamed: 0.1", axis=1, inplace=True)
    
    data = data[data["Season"] == 2022]
    data.index.rename("ID", inplace=True)
    data.rename(columns={"Tm":"team"}, inplace=True)
    
    def create_per_game(col):
        
        data[f"{col}_per_game"] = data[col]/(data["MP"])
        
        return data[f"{col}_per_game"]
    
    convert_list = ['ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', "FG%", "3P%"]
    
    for col in convert_list[-2:]:
        data[col].fillna(0, inplace=True)
    
    for col in convert_list:
        create_per_game(col) 
    
    #deal with duplicate players
    data = data.sort_values(by="MP", ascending=False).drop_duplicates(subset="Player", keep="first")
        
    player_data = data.copy()
    position = list(player_data["Pos"].unique())[:5]
    
    #creating player rating column
    rating = []

    for player_index in range(len(player_data)):

        player = player_data.iloc[player_index]
        player_position = player[2]

        if player_position in position:
            player_position = player_position
        elif player_position == 'C-PF':
            player_position = player[2][:1]
        else:
            player_position = player[2][:2] 

        #calculaste player rating adding up  the scores of the last columns after dividing by the max
        #score of players in that positions
        max_orb = player_data[player_data["Pos"] == player_position]['ORB_per_game'].max()
        max_drb = player_data[player_data["Pos"] == player_position]['DRB_per_game'].max()
        max_trb = player_data[player_data["Pos"] == player_position]['TRB_per_game'].max()
        max_ast = player_data[player_data["Pos"] == player_position]['AST_per_game'].max()
        max_stl = player_data[player_data["Pos"] == player_position]['STL_per_game'].max()
        max_blk = player_data[player_data["Pos"] == player_position]['BLK_per_game'].max()
        max_tov = player_data[player_data["Pos"] == player_position]['TOV_per_game'].max()
        max_pf = player_data[player_data["Pos"] == player_position]['PF_per_game'].max()
        max_pts = player_data[player_data["Pos"] == player_position]['PTS_per_game'].max()
        max_fg = player_data[player_data["Pos"] == player_position]['FG%_per_game'].max()
        max_3pt = player_data[player_data["Pos"] == player_position]['3P%_per_game'].max()
        max_gs = player_data[player_data["Pos"] == player_position]["G"].max()

        ratings = (((player["ORB_per_game"]/max_orb) + (player["DRB_per_game"]/max_drb) + (player["TRB_per_game"]/max_trb) + (player["AST_per_game"]/max_ast) +\
        (player["STL_per_game"]/max_stl) +(player["BLK_per_game"]/max_blk) + (player["TOV_per_game"]/max_tov) + (player["PF_per_game"]/max_pf) +\
        (player["PTS_per_game"]/max_pts) + (round(player["FG%_per_game"]/max_fg, 3)) + (round(player["3P%_per_game"]/max_3pt, 3)) )*100)/9
        
        if player["MP"] > 800:
            if (player["MP"] > 0.6 * (player["G"] * 45)) & (player["MP"] > 1000):
                ratings += 35
            elif (player["MP"] > 0.5 * (player["G"] * 45)) & (player["MP"] > 600):
                ratings += 30
            elif player["MP"] < 300:
                ratings -= 20
            else:
                ratings += 25
        else:
            ratings += 5
            
        if (ratings > 65) & (player["MP"] < 600):
            ratings -= 15
        
        if player["MP"] > 800:
            if ((player['3P%_per_game'] > 0.8 * max_3pt) | (player['FG%_per_game'] > 0.8 * max_fg)) & (ratings < 80):
                ratings += 20
                
        if player["3P"] > 1000:      
            ratings += 10
        elif player["3P"] > 1600:
            ratings += 20
        else:
            ratings += 5
            
        if player["PTS"] < 400:
            ratings = 50 + (ratings * 0.025)
            
        #improving ratings based on score points
        if (player["Pos"] == "PF") | (player["Pos"] == "SF")  | (player["Pos"] == "SG"):
            if player["PTS"] > 1600:
                ratings += 15
            elif player["PTS"] > 1000:
                ratings += 10
            else:
                ratings += 5
                
        elif (player["Pos"] == "PG"):
            if player["PTS"] > 1600:
                ratings += 4
            elif player["PTS"] > 1000:
                ratings += 3
            else:
                ratings += 2
                
        if (player["Pos"] == "C"):
            if player["PTS"] > 1600:
                ratings += 5
            elif player["PTS"] > 1000:
                ratings += 4
            else:
                ratings += 3
        
        else:
            if player["PTS"] > 1600:
                ratings += 20
            elif player["PTS"] > 1000:
                ratings += 15
            else:
                ratings += 10
                
        rating.append(ratings)
        for i in range(len(rating)):
            if rating[i] > 100:
                rating[i] = 100
            
    player_data["Rating"] = rating
    player_data.Rating = player_data["Rating"].astype(int)
        

    return data, player_data



def team_finder(player_data):
    """
    saves the .csv files of individual teams to squad directory
    -inputs: collects the player data, type: pd.DataFrame
    
    """
    teams = list(player_data.team.unique())
    
    def find_squad(team):
        team_player_data = player_data[player_data.team == team]
        #nop = len(team_player_data)

        return team_player_data
    
    for t in teams:
        team_squad = find_squad(t)
        team_squad.to_csv(f"C:/Users/DELL/Project/FreeWorks/NBA  Project/data/squad_data/{t.lower()}-squad.csv")
        
        
        
def show_distribution(var, name):
    """The functions helps to print out statistical analysis about data and then plots a distribution plot for the data 
    - inputs: It takes in the column to be analysed
    - Returns statistical analyses and visuals for the data"""
    
    sns.set()
    
    #collating the statistical property
    min_val = var.min()
    max_val = var.max()
    mode_val = var.mode()[0]
    median_val = var.median()
    mean_val = var.mean()
    rstd = var.std()
    rvar= np.sqrt(rstd)
    
    print("The statistical values are as follows:\n Minimum value:{:.2f}\n Maximum value: {:.2f}\n Mode: {:.2f} \n Mean: {:.2f} \n Standard Deviation: {:.2f} \n Variance: {:.2f} \n Median: {:.2f}"\
          .format(min_val, max_val, mode_val, mean_val, rstd, rvar, median_val))
    
    #creating a figure with 2 rows and 1 columns where the upperpart is dominated by the histplot and the
    #bottom by the 
    fig, ax = plt.subplots(2,1, figsize=(12,6))
    
    #Creating the histogram
    sns.histplot(var, ax=ax[0])
    ax[0].set_ylabel("Frequency")
    
    #fitting the statiscal lines
    ax[0].axvline(x=min_val, color='gray', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=median_val, color='cyan', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=mode_val, color='red', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=mean_val, color='orange', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=rvar, color='white', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=max_val, color='gray', linestyle='dashed', linewidth=2)
    
    #creating the boxplot
    sns.boxplot(x=var, ax=ax[1])
    ax[1].axvline(x=min_val, color='gray', linestyle='dashed', linewidth=2, label="Minimum Value")
    ax[1].axvline(x=median_val, color='cyan', linestyle='dashed', linewidth=2, label="Median")
    ax[1].axvline(x=mode_val, color='red', linestyle='dashed', linewidth=2, label="Mode")
    ax[1].axvline(x=mean_val, color='orange', linestyle='dashed', linewidth=2, label="Mean")
    ax[1].axvline(x=rvar, color='white', linestyle='dashed', linewidth=2, label="Variance")
    ax[1].axvline(x=max_val, color='gray', linestyle='dashed', linewidth=2, label="Standard Deviation")
    
    fig.suptitle(f"Data Distribution for {name}", color="brown")
    plt.legend()
    
    plt.show()

    
def create_conference(data):
    """
    Helps create conference column for each of the teams
    """
    west = ['DAL', 'DEN', 'GSW', 'HOU', 'LAC', 'LAL', 'MEM', 'MIN', 'NOP', 'OKC', 'POR', 'PHO', 'SAC', 'SAS', 'UTA']
    category = []
    wining = []
    for col in range(len(data)):
        wining_rate = (data.iloc[col][-5]/(data.iloc[col][-4] + data.iloc[col][-5])) * 100
        wining.append(wining_rate)
        
        if data.iloc[col][1] in west:
            category.append("W")
        else:
            category.append("E")
    data["win_rate"] = wining
    data["Conference"] = category
    
    data = data[['team', 'Alias', "Conference", 'score1', 'score2', 'win', 'loss','win_prob_avg', "win_rate", 'elo1_pre',
                 'elo2_pre', 'elo_prob1','elo_prob2', 'elo1_post', 'elo2_post', 'raptor1_pre', 'raptor2_pre',
                 'raptor_prob1','raptor_prob2']]
    
    return data
    




