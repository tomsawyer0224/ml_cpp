/*
void LogisticRegression::fit(const arma::mat& X_train, const arma::mat& y_train){
	arma::uword N = X_train.n_rows;
	arma::mat one = arma::mat(N, 1, arma::fill::ones);
	arma::mat X = arma::join_rows(one, X_train);
	
	std::string optimizer = this->params_string["optimizer"];
	if(optimizer == "none"){
		SGD solver(this->params_double["learning_rate"]);
		//this->_fit<SGD>(X, y_train, solver);
		this->_fit_SGD<SGD>(X, y_train, solver);
	}
	if(optimizer == "momentum"){
		SGDmomentum solver(this->params_double["learning_rate"], this->params_double["momentum"]);
		//this->_fit<SGDmomentum>(X, y_train, solver);
		this->_fit_SGD<SGDmomentum>(X, y_train, solver);
	}
}
*/


void LogisticRegression::fit(const arma::mat& X_train, const arma::mat& y_train){
	arma::uword N = X_train.n_rows;
	arma::mat one = arma::mat(N, 1, arma::fill::ones);
	arma::mat X = arma::join_rows(one, X_train);
	
	std::string optimizer = this->params_string["optimizer"];
	arma::mat coef(X.n_cols, 1);//initialize coef
	if(optimizer == "none"){
		SGD solver(this->params_double["learning_rate"]);
		//this->_fit<SGD>(X, y_train, solver);
		this->_fit_SGD<SGD>(coef, X, y_train, solver);
	}
	if(optimizer == "momentum"){
		SGDmomentum solver(this->params_double["learning_rate"], this->params_double["momentum"]);
		//this->_fit<SGDmomentum>(X, y_train, solver);
		this->_fit_SGD<SGDmomentum>(coef, X, y_train, solver);
	}
}




arma::mat LogisticRegression::predict_proba(const arma::mat& X_test){
	arma::uword N = X_test.n_rows;
	arma::mat one = arma::mat(N, 1, arma::fill::ones);
	arma::mat X = arma::join_rows(one, X_test);
	
	arma::mat z = X*this->params_mat["coef"];//matrix n_samples x 1
	arma::mat a = sigmoid(z);//matrix n_samples x 1
	
	//arma::umat ua = arma::conv_to<arma::umat>::from(a);
	return a;
}

arma::mat LogisticRegression::predict(const arma::mat& X_test){
	arma::mat a = this->predict_proba(X_test);
	arma::mat y_pred(a.n_elem,1);
	for(arma::uword i = 0; i < a.n_elem; i++){
		if(a[i] < 0.5){
			y_pred[i] = 0;
		}
		else{
			y_pred[i] = 1;
		}
	}
	return y_pred;
}

double LogisticRegression::score(const arma::mat& X, const arma::mat& y){;
	return accuracy_score(y, this->predict(X));
}

arma::mat LogisticRegression::gradient(const arma::mat& theta, const arma::mat& X, const arma::mat& y){
	arma::mat z = X*theta;//matrix n_samples x 1
	arma::mat a = sigmoid(z);//matrix n_samples x 1
	return X.t()*(a-y);//matrix n_features x 1
}
/*
template <class OPT>
void LogisticRegression::_fit(const arma::mat& X, const arma::mat& y, OPT& solver){
	arma::uword n_samples = X.n_rows;
	arma::uword n_features = X.n_cols;
	
	arma::mat coef(n_features,1);
	arma::mat coef_last(coef);
	for(int iter = 0; iter < this->params_int["max_iter"]; iter++){
		arma::uvec index = arma::randperm(n_samples);
		for(arma::uword i = 0; i < n_samples; i++){
			arma::mat Xi = X.submat(index[i],0,index[i],n_features-1); 
			arma::mat yi = y.submat(index[i],0,index[i],0);
			arma::mat grad = this->gradient(coef, Xi, yi);
			solver.update(coef, grad);
			if(arma::norm(coef - coef_last, "fro") < this->params_double["tol"]){
				this->set_params("coef", coef);
				this->set_params("n_iters", iter+1);
				//std::cout<<"n_iter = "<<iter<<std::endl;
				return;
			}
			coef_last = coef;
		}
	}
	this->set_params("coef", coef);
	this->set_params("n_iters", this->params_int["max_iter"]);
}
*/

