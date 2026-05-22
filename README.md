# Shooters Shoot - A Shot Analysis Model
## Introduction
I started this project in order to better understand the data libraries pandas, xgboost and scikit-learn, and the matplotlib graphics package.
The goal was to create a functioning regression model that takes in the record of a basketball player's shots as data in order to train itself to spot where said player is most likely (or least likely) to make a shot.
The model then outputs two images, one shot chart that shows made and missed shots, and a heatmap that shows which shots are most likely to go in the basket.
The data I used for this project was scraped from nba_api. I also got inspiration from this Github project: ([https://github.com/slieb74/NBA-Shot-Analysis])

## Testing Stage
I wanted to start this project with some tests, to fully understand the scope of the API I was working with. 
My first instinct was to see how many players were in the database, and also to check for duplicates. 
I also did some tweaking to make sure the user can access a team's record and performance in a given season, or in the playoffs if they wish. 
Finally, I wanted to explore the extent of the 'shotchartdetail' module by adding everything I've experimented with combined with the attributes of the module to give a conclusive overview of shots taken in basketball games across the NBA.

## Data Collection
For this purpose, I'm using a notebook to demonstrate data collection into a .parquet file (for when I need a large sample size of data) and a file of helper functions I'll be using in the main notebook.
This .py file has basic functionalities, namely;
-Getting the ID of a player or a team (We'll use this to precisely access their shot data)
-Correcting any errors the user may have typed in by using a format helper function 
-Fetching shot data from the API using its built-in protocols

## Feature Engineering
This section also has a notebook and a .py file, though I was only using the notebook to test my logic.
The functionalities of the .py file are as follows;
-The total seconds remaining on the clock are counted, excluding overtime (I can handle this in a later commit if I need to, but for now I'm looking at regulation time).
-The binary counter that shows if the player made the shot is set as the 'target'.
-The code 'engineers' the columns we need and returns a tuple for our model to train itself with.

## Model Training
This notebook and .py file trains a simple regression model to use our filtered data to produce;
- A mean accuracy score that compares our real shot that our player made (or missed) to the data-backed prediction that we have on said player
- A classification report that measures the precision of our prediction concerning our data
- A confusion matrix that looks for Type I and Type II errors in our prediction model 
    - (for reference; Type 1 Error: negative statistic perceived as positive, Type 2 Error: positive statistic perceived as negative)
- A log-loss score that reduces as our model is fed more data
- A Brier Score that measures the mean difference between our predicted outcome and our actual outcome
- A ROC-AUC Score that evaluates our model's ability to differentiate between a positive outcome and a negative outcome

After which, the model is saved as a .json file for the Main notebook to analyze and produce legible data.

## Analysis and Visualization
Now we are at the final section, where we do visual analysis. I wrote my court visualizations on a different .py file before combining all the functions I'd used in the other notebooks to create a sample analysis.
The images are then saved in a seperate folder.
