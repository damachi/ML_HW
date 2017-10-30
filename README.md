#Classify Higgs Boson using Machine Learning

The "run.py" allow user to classify the Higgs boson based on its decay signature.
The program create a file "prediction_1.csv" representing the prediction of the test dataset, ready to be submitted on kaggle.

##Installation

The machine that will run "run.py" should have installed:
		- python3 https://www.python.org/downloads/
		- the numpy library  http://www.numpy.org/

##Running the code

Run the following command in your terminal inside the folder Higgs_Boson:

$ python3 run.py

##Folder organization

### - cost.py
	Contains the costs functions methods:
		compute_mse(y, tx, w)
		calculate_loss(y, tx, w, lambda_=0)

### -features_processing.py
	Contains the methods used in the feature processing process:
		nan_handler(tx)
		build_model_data(y, x)
		build_poly2_matrix(x)
		discrete_to_cat(x, cat_index)
		angles_to_sin_cos(tx, norm_and_angles_features_indices)
		remove_outliers(y, tx, cat, phi_angles)
		correlated_features(x, coef)
		standardize(x)
		standardize_test(x_tr, x_te)
### -gradients.py
	Contains the gradients(/hessian) methods :
		compute_gradient_mse(y, tx, w)
		compute_gradient_likelihood(y, tx, w, lamdba_=0)
		double_pen_gradient_likelihood(y, tx, w, lambda_)
		calculate_hessian(y, tx, w)
### -hw_helpers.py
	Contains some helpers methods that were implemented during our lab sessions:
		batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)
		sigmoid(t)
		build_poly(x, degree)
		absolute_error(y, y_pd)
### -implementations.py
	Contains the differents regression methods:
		least_squares_GD(y, tx, initial_w, max_iters, gamma)
		least_squares_SDG(y, tx, initial_w, max_iters, gamma)
		least_squares(y,tx)
		ridge_regression(y, tx, lambda_)
		logistic_regression(y, tx, initial_w, max_iters, gamma)
		reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
		new_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
		double_pen_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)

For complementary informations about the methods check the DocStrings
