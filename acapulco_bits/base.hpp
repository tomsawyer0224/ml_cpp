//-------------------------Base-------------------------------------
void Base::get_params(){
	for(auto it = this->params_double.begin(); it != this->params_double.end(); it++){
		std::cout<<it->first<<": "<<it->second<<std::endl;
	}
	for(auto it = this->params_string.begin(); it != this->params_string.end(); it++){
		std::cout<<it->first<<": "<<it->second<<std::endl;
	}
	for(auto it = this->params_int.begin(); it != this->params_int.end(); it++){
		std::cout<<it->first<<": "<<it->second<<std::endl;
	}
	for(auto it = this->params_bool.begin(); it != this->params_bool.end(); it++){
		std::cout<<it->first<<": "<<it->second<<std::endl;
	}
	for(auto it = this->params_mat.begin(); it != this->params_mat.end(); it++){
		it->second.print(it->first + ":");
	}
}

double Base::get_params_double(const std::string& p){
	if(this->params_double.count(p) == 0){
		std::cout<<"get params failed! parameter '"<<p<<"' not found"<<std::endl;
		exit(1);
	}
	return this->params_double[p];
}
std::string Base::get_params_string(const std::string& p){
	if(this->params_string.count(p) == 0){
		std::cout<<"get params failed! parameter '"<<p<<"' not found"<<std::endl;
		exit(1);
	}
	return this->params_string[p];
}
int Base::get_params_int(const std::string& p){
	if(this->params_int.count(p) == 0){
		std::cout<<"get params failed! parameter '"<<p<<"' not found"<<std::endl;
		exit(1);
	}
	return this->params_int[p];
}
bool Base::get_params_bool(const std::string& p){
	if(this->params_bool.count(p) == 0){
		std::cout<<"get params failed! parameter '"<<p<<"' not found"<<std::endl;
		exit(1);
	}
	return this->params_bool[p];
}

arma::mat Base::get_params_mat(const std::string& p){
	if(this->params_mat.count(p) == 0){
		std::cout<<"get params failed! parameter '"<<p<<"' not found"<<std::endl;
		exit(1);
	}
	return this->params_mat[p];
}


void Base::set_params(const std::string& p, const double& val){
	this->params_double[p] = val;
}

void Base::set_params(const std::string& p, const char* val){
	this->params_string[p] = val;
}

void Base::set_params(const std::string& p, const int& val){
	this->params_int[p] = val;
}
void Base::set_params(const std::string& p, const bool& val){
	this->params_bool[p] = val;
}
void Base::set_params(const std::string& p, const arma::mat& val){
	this->params_mat[p] = val;
}


//-------------------------BaseOptimizer-------------------------------------

BaseOptimizer::BaseOptimizer(double learning_rate = 0.01){
	this->params_double["learning_rate"] = learning_rate;
}

void BaseOptimizer::update(arma::mat& theta, const arma::mat& grad){
//updates theta(include weights and biases from gradient) 
	arma::mat upd = this->get_update(grad);
	theta = theta + upd;
}


//-------------------------BaseModel-------------------------------------

BaseModel::BaseModel(){
	this->params_double["learning_rate"] = 0.01;
	this->params_int["max_iter"] = 200;
	this->params_string["optimizer"] = "none";
	this->params_double["tol"] = 0.0001;
	this->params_double["momentum"] = 0.9;
}
/*
template <class OPT>
void BaseModel::_fit_SGD(const arma::mat& X, const arma::mat& y, OPT& solver){
	arma::uword n_samples = X.n_rows;
	arma::uword n_features = X.n_cols;
	arma::uword n_features_y = y.n_cols;//n_classes
	//arma::uword n_classes = y.n_cols;//n_classes
	
	arma::mat coef(n_features,1);
	arma::mat coef_last(coef);
	for(int iter = 0; iter < this->params_int["max_iter"]; iter++){
		arma::uvec index = arma::randperm(n_samples);
		for(arma::uword i = 0; i < n_samples; i++){
			arma::mat Xi = X.submat(index[i],0,index[i],n_features-1); 
			arma::mat yi = y.submat(index[i],0,index[i],n_features_y-1);
			//arma::mat yi = y.submat(index[i],0,index[i],n_classes-1);
			arma::mat grad = this->gradient(coef, Xi, yi);
			solver.update(coef, grad);
			if(arma::norm(coef - coef_last, "fro") < this->params_double["tol"]){
				this->set_params("coef", coef);
				this->set_params("n_iters", iter+1);
				return;
			}
			coef_last = coef;
		}
	}
	this->set_params("coef", coef);
	this->set_params("n_iters", this->params_int["max_iter"]);
}
*/

template <class OPT>
void BaseModel::_fit_SGD(arma::mat& theta, const arma::mat& X, const arma::mat& y, OPT& solver){
	arma::uword Xy_n_rows = X.n_rows, X_n_cols = X.n_cols, y_n_cols = y.n_cols;
	arma::mat theta_last(theta);
	for(int iter = 0; iter < this->params_int["max_iter"]; iter++){
		arma::uvec index = arma::randperm(Xy_n_rows);
		for(arma::uword i = 0; i < Xy_n_rows; i++){
			arma::mat Xi = X.submat(index[i],0,index[i],X_n_cols-1); 
			arma::mat yi = y.submat(index[i],0,index[i],y_n_cols-1);
			arma::mat grad = this->gradient(theta, Xi, yi);
			solver.update(theta, grad);
			//check stopping criteria
			if(arma::norm(theta - theta_last, "fro") < this->params_double["tol"]){
				this->set_params("coef", theta);
				this->set_params("n_iters", iter+1);
				return;
			}
		}
	}
	theta_last = theta;
	this->set_params("coef", theta);
	this->set_params("n_iters", params_int["max_iter"]);
	return;
}
//-------------------------BaseRegression-------------------------------------


