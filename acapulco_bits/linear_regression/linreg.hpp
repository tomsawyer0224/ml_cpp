void LinearRegression::fit(const arma::mat& X_train, const arma::mat& y_train){
	arma::uword N = X_train.n_rows;
	arma::mat one = arma::mat(N, 1, arma::fill::ones);
	arma::mat X = arma::join_rows(one, X_train);
	arma::mat coef = arma::pinv((X.t()*X))*X.t()*y_train;
	this->set_params("coef", coef);
}

arma::mat LinearRegression::predict(const arma::mat& X_test){
	arma::uword N = X_test.n_rows;
	arma::mat one = arma::mat(N, 1, arma::fill::ones);
	arma::mat X = arma::join_rows(one, X_test);
	return X*this->params_mat["coef"];
}

double LinearRegression::score(const arma::mat& X, const arma::mat& y){
	arma::mat y_pred = this->predict(X);
	return r2_score(y, y_pred);
}

