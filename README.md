# uu-ds-datamining-assignment

## Install Dependencies
`$ pip install -r requirements.txt`

## Implementation of Logistic Regression and Naive Bayes
 <p>
    Naive Bayes is executed in `run_MultiNB_logReg.py`. Its only tunable parameter `alpha` is an array computed from the training data beforehand. The scripts runs multiple iterations with different configurations of number of splits and the max number of features BoW should consider ranging in the values {5,10} and {150,200,250,300,350,400,450,500,550,600}.
 </p>
 <p>
    Logistic Regression is executed in the same file the same way, with the addition of grid search cv to find the best configuration for `C` and `solver`
 </p>
