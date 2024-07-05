
void SoftmaxRegression::fit(const arma::mat& X_train, const arma::mat& y_train){
	arma::uword N = X_train.n_rows;
	arma::mat one = arma::mat(N, 1, arma::fill::ones);
	arma::mat X = arma::join_rows(one, X_train);
	arma::mat y = one_hot_encoder(y_train);
	
	std::string optimizer = this->params_string["optimizer"];
	arma::mat coef(X.n_cols, y.n_cols);//initialize coef (size n_features x n_classes)
	if(optimizer == "none"){
		SGD solver(this->params_double["learning_rate"]);
		this->_fit_SGD<SGD>(coef, X, y, solver);
	}
	if(optimizer == "momentum"){
		SGDmomentum solver(this->params_double["learning_rate"], this->params_double["momentum"]);
		this->_fit_SGD<SGDmomentum>(coef, X, y, solver);
	}
}

arma::mat SoftmaxRegression::predict_proba(const arma::mat& X_test){
	arma::uword N = X_test.n_rows;
	arma::mat one = arma::mat(N, 1, arma::fill::ones);
	arma::mat X = arma::join_rows(one, X_test);
	//return X*this->params_mat["coef"];
	arma::mat Z = X*this->params_mat["coef"];
	arma::mat A = softmax(Z,1);
	return A;
}

arma::mat SoftmaxRegression::predict(const arma::mat& X_test){
	arma::mat Y_proba = this->predict_proba(X_test);
	arma::umat y_pred_int = arma::index_max(Y_proba, 1);
	arma::mat y_pred = arma::conv_to<arma::mat>::from(y_pred_int);
	return y_pred;
}

double SoftmaxRegression::score(const arma::mat& X, const arma::mat& y){;
	arma::mat y_pred = this->predict(X);
	return accuracy_score(y, y_pred);
}

arma::mat SoftmaxRegression::gradient(const arma::mat& theta, const arma::mat& X, const arma::mat& y){
	arma::mat Z = X*theta;
	arma::mat A = softmax(Z, 1);
	arma::mat E = A - y;
	return X.t()*E;
}


