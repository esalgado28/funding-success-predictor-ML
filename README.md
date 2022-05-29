# deep-learning-challenge
Eddy' folder for Deep Learning Homework
## Overview
A non-profit, Alphabet Soup, wants to create an algorithm to predict whether or not applicants for funding will be successful based on metadata for each organization. To help them, we will utilize machine learning, specifically neural networks with multiple hidden layers. The data is provided to us in `charity_data.csv` which contains over 34,000 applicants that have recieved funding in the past. Our goal is to build a model with over 75% prediction accuracy.

## Results
- Data Preprocessing
  - The target variable for our model is `IS_SUCCESSFUL` which denotes whether the funding was successful as either a 1 or 0.
  - All other variables asides from `EIN` and `NAME` were used as features in the first model. In the optimized model, `NAME` was used as a feature while `STATUS` and `SPECIAL_CONSIDERATIONS` were dropped due to the rarity of data with different values.
  - `EIN` was removed from the input data in both cases as it is a purely artificial identifier so should not have any use in the model.
  - Categorical data was converted using `pd.get_dummies` and all data was then scaled using a standard scaler.

- Compiling, Training, and Evaluating the Model
  - The neural network consists of an input layer which takes 259 inputs, followed by 3 hidden layers with 128, 64, and 32 nodes, respectively. The output layer consists of a single node since the model serves as a binary classifier.
  - The choices for the number of nodes follows the rule of thumb of having less nodes than features and each subsequent layer having less nodes. I also chose nice numbers that are powers of 2 which is not necessary and otherwise arbritrary.
  - The initial model achieved an accuracy of 73% but after some optimization, the accuracy was bumped up to about 78% which surpassed our target.
  - To improve the model, the `STATUS ` and `SPECIAL_CONSIDERATIONS` features were dropped and the `NAME` column was added back into the input data. The `NAME` column turned out to be a useful feature as there were many organizations that received a significant portion of the granted funding requests. Secondly, I added bins to `AFFILIATION` and `ORGANIZATION` to consolidate the rare options. Finally, I chose to add a third layer and more neurons due to the increase in feature space after adding `NAME` and I used kerastuner to further tune the hyperparameters.

## Summary
The deep learning model was successful in surpassing the target accuracy with an accuracy of 78%. The model was improved by implementing some extra preprocessing steps and tweaking the hyperparameters of the neural network. The main contributor to the performance increase, however, seemed to be the addition of the `NAME` column. This was not obvious at first as this seemed more like an identifier than a useful feature. Various hyperparameter configurations were tried but only resulted in marginal changes to the accuracy. This highlights the importance of data exploration and preprocessing. Other models besides a neural network that I would recommend exploring would be a random forest classifier and logistic regression as those models can also take a large amount of features and output a binary classification.
