import pandas as pd
import json
import os
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

#Filter some warnings given by sklearn during fitting
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def initialize_models():
	"""
	Initialize a dictionary with the model objects to be used
	inputs
		-
	outputs
		-
	"""
    
    linear_classifier = SGDClassifier(loss="modified_huber", penalty="l2", warm_start=True)
    random_forest_classifier = RandomForestClassifier(warm_start=True)
    
    return {"random_forest": random_forest_classifier, 'linear_classifier': linear_classifier}
	
def fit_model(model, df, target = 'conversions'):
    """
	Fit the model(s) to the observations
	inputs
		model - sklearn model object or dict with sklearn model objects as values
		df - pd.DataFrame, training data
		target - str, name of target feature
	outputs
		model - sklearn model object or dict with sklearn model objects as values fitted to the training data
	"""
	
    print("Fitting the model to the observations...")
    # We only will predict on valid click events 
    df = df[df['valid_clicks'] == 1]
    # Balance the classes by downsampling, we have plenty of records
    df_conversions = df[df[target] == 1]
    df_no_conversion = df[df[target] == 0].sample(len(df_conversions))
    
    # Combine the positive and negative records, and shuffle them
    df_train = pd.concat([df_conversions, df_no_conversion], axis=0).sample(frac=1)
    
    # Separate the features from the target
    X = df_train[[feature for feature in list(df_train) if '_rate_past' in feature]]
    y = df_train[target]
    
    # Fit and return the models
    
    # If passing a dictionary of model objects, fit all of them
    if isinstance(model, dict):
        for model_name, model_object in model.items():
            print("Fitting the %s" % model_name)
            model_object.fit(X, y)
    # Otherwise, just fit the model that was passed
    else:
        model.fit(X, y)
    
    return model
	
def make_test_predictions(model, df, target = 'conversions'):
    """
	inputs
		model - sklearn model object
		df - pd.DataFrame, testing data
		target - str, name of target
	outputs
		y - pd.Series, ground truth values
		class_predictions - numpy 1-D array, class predictions (0, 1)
		class_probability_estimates - numpy 1-D array, class probablity estimates
	"""
	
    X = df[[feature for feature in list(df) if '_rate_past' in feature]]
    y = df[target]
	
	class_predictions = model.predict(X)
	class_probability_estimates = model.predict_proba(X)[:,1]
    
    return y, class_predictions, class_probability_estimates

def train_and_predict(processed_csv_filenames_list, days_to_train, days_to_predict):
    """
    Train the model on a given number of days, 
    predict on a hold out test set of the days following the training set,
    save the predictions and class probabilities to disk as a JSON for further evaluation
	
    inputs
        processed_csv_filenames_list - list, names of processed csv files
        days_to_train - int, number of days to train on
        days_to_predict - int, number of days to predict on
    outputs
        predictions_dict - dict, dictionary with one key for each training day,
                           containing the ground truth values of the test set, 
                           and the class prediction and probabilities of each fitted model
    """
    
    # If the function is called with more days to train than would leave a 3 day test partition, set the values manually
    if days_to_train >= (len(processed_csv_filenames_list)-1) - days_to_predict:
        days_to_predict = 3
        days_to_train = (len(processed_csv_filenames_list)-1) - days_to_predict
    # Initialize the models, in this case a linear classifier and a random forest
    model_dict = initialize_models()
    # Initialize a predictions_dict
    predictions_dict = {}
    #We will evalaluate
    test_days = processed_csv_filenames_list[len(processed_csv_filenames_list)-3:]
    # Loop over the training days, read the data, and fit the model
    # Skip the first day, because the cumulative features may not be well established
    for i in range(1, days_to_train+1):
        print("Training on day %d of %d" % (i, days_to_train))
        df_train = pd.read_csv(os.getcwd() + '/processed_csv/' + processed_csv_filenames_list[i])
        
        print("Working with csv %s" % processed_csv_filenames_list[i])
        #Fit the linear classifier
        model_dict = fit_model(model_dict, df_train)
        
        # By default, we predict on the number of days given by days_to_predict
        # but if the training set size is less than 5 days, fit on only 1 day; 2 days if less than 10
        if i < 5:
            prediction_days = min(1, days_to_predict)
        elif i < 10:
            prediction_days = min(2, days_to_predict)
        else:
            prediction_days = days_to_predict
            
        # Loop over the prediction days, read the data, and save the predictions in a dictionary
        
        # Initialize a key in the prediction dict for the number of training days
        predictions_dict['training_days_%d' % i] = {}
        predictions_dict['training_days_%d' % i]['ground_truth'] = []
        for model_object_name in model_dict.keys():
            predictions_dict['training_days_%d' % i][model_object_name] = {}
            predictions_dict['training_days_%d' % i][model_object_name]['predictions'] = []
            predictions_dict['training_days_%d' % i][model_object_name]['probabilities'] = []

        # Predict from the following day up to the number of prediction days
        for j in range(i+1, (i + 1) + prediction_days):
            print("*" * 3 + " Predicting on day %d with %d training days" % (j, i))
            df_predict = pd.read_csv(os.getcwd() + '/processed_csv/' + processed_csv_filenames_list[j])
            print("*" * 3 + " Working with csv %s" % processed_csv_filenames_list[j])
            df_predict[df_predict['valid_clicks'] == 1]
            predictions_dict['training_days_%d' % i]['ground_truth'] += list(df_predict['conversions'])     

            for model_object_name, model_object in model_dict.items():
                print("*" * 6 + " Predicting with the %s" % model_object_name)
                _, prediction_values, prediction_probabilities = make_test_predictions(model_object, df_predict)

                predictions_dict['training_days_%d' % i][model_object_name]['predictions'] += list(prediction_values)
                predictions_dict['training_days_%d' % i][model_object_name]['probabilities'] += list(prediction_probabilities)
        with open('predictions/prediction_eval_training_days_%d.json' % i, 'w') as f:
            predictions_json = json.dumps(predictions_dict['training_days_%d' % i])
            f.write(predictions_json)
    return predictions_dict