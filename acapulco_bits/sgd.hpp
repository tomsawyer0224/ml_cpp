//------------------SGD------------------------------------
SGD::SGD(double learning_rate = 0.01):BaseOptimizer::BaseOptimizer(learning_rate){

}

arma::mat SGD::get_update(const arma::mat& grad){
	double learning_rate = this->params_double["learning_rate"];
	return -learning_rate*grad;
}

//------------------SGDmomentum------------------------------------
SGDmomentum::SGDmomentum(double learning_rate = 0.01, double momentum = 0.9):BaseOptimizer::BaseOptimizer(learning_rate){
	this->params_double["momentum"] = momentum;
}

arma::mat SGDmomentum::get_update(const arma::mat& grad){
	if(this->params_mat.count("velocity") == 0){
		arma::mat v_init(grad.n_rows, grad.n_cols);
		//this->set_params("velocity", v_init);
		this->params_mat["velocity"] = v_init;
	}
	arma::mat& velocity = this->params_mat["velocity"];
	double learning_rate = this->params_double["learning_rate"];
	double momentum = this->params_double["momentum"];
	velocity = momentum*velocity + learning_rate*grad;
	return -velocity;
}
