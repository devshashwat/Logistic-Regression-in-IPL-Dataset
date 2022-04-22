# Logistic-Regression-in-IPL-Dataset

 Predict the match winning outcome of Dhoni for CSK. Further conditions: 
 • It is given that Dhoni has to play the last over is not dismissed given that the last over has to be of the second innings.
 
My Approach and Code:

Why Logistic Regression?

The model should be able to predict the dependent variable as one of the two probable class which could be 0 or 1. 
If we consider using Linear Regression, we can predict the value for the given set of rules as input to the modelbut it will forecast continuous values like 0.03, +1.2, -0.9, etc. which is not suitable for categorizing it in one of the two classes neither identifying it as a probability value since we have to decide,
whether the team won the match or not using dependent variables.
This can also help in finding the accuracy of the prediction and it is widely used in machine learning as well.

• First of all, for the logistic Regression, the independent values were decided to be the ones shown in the code which are:
1. Second innings played
2. Last over played
3. Not dismissed in last over

• Furthermore, now all the required values for the LR were converted into new lists which would be later used.

• Total Matches played by the player is used as the base of this new list.

Now, list of matches won is chosen as the dependent variable and is also converted into a list.

Now, here we compare all the independent variable list with the on in dependent and create a new list out of it where the output will 0 or 1.

• Through this we can have new list which is not included in the initial given database, of if player played in last over, or second innings or that if he was dismissed in the last over.

• This also allows us the give the output of match winner in 0 and 1 format.

• For all the above cases, 1 is true and 0 is false.

• To do that for loop was used until the final value of matches played and then value was assigned to a new list according to the position of the initial data frame in the new list.

• The values being 0 and 1. The next step would be merging the list into one new data frame as will be shown below.

• Now in this code, using dictionary we combine the given list then convert the given dictionary into the data frame which we will use for LR.

• Now, we have a ideal data frame table required for the logistic regression.

• Now, in this phase we initialize the logistical regression by defining the independent and dependent values as X and Y respectively.

• Then we move on to set the train and test data ratio which is set to 75% and 25% respectively.

• Then we apply the logistic regression algorithm accordingly with already defined functions in python and find the predicted value of the text data in y prediction.

Finally, here we display the accuracy of the outcome and also plot the confusion matrix by the given functions in python which were imported before.
