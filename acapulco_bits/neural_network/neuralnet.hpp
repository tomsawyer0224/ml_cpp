
MLP::MLP(const std::vector<int>& layers):BaseModel::BaseModel(){
	this->params_string["activation"] = "sigmoid";
	//this->params_bool["early_stopping"] = true;
	this->layers = layers;
	
	/*
	int n_layers = layers.size();
	this->coefs.set_size(n_layers - 1);
	this->intercepts.set_size(n_layers - 1);
	for(int i = 0; i < n_layers - 1; i++){
		//arma::mat coef(layers[i+1], layers[i]);
		arma::mat coef(layers[i+1], layers[i], arma::fill::randu);
		coefs(i) = coef;
		//arma::mat intercept(layers[i+1], 1);
		//arma::mat intercept(layers[i+1], 1, arma::fill::randu);
		arma::mat intercept(layers[i+1], 1, arma::fill::zeros);
		intercepts(i) = intercept;
	}
	*/
	//pkg.unpack(this->coefs, this->intercepts);
}

void MLP::init_coefs_inters(arma::field<arma::mat>& coefs, arma::field<arma::mat>& intercepts){
	int n_coefs = coefs.n_elem, n_intercepts = intercepts.n_elem;
	for(int i = 0; i < n_coefs; i++){
		coefs(i).ones();
	}
	for(int i = 0; i < n_intercepts; i++){
		intercepts(i).ones();
	}
	this->coefs = coefs;
	this->intercepts = intercepts;
}


void MLP::get_params(){
	std::cout<<"layers: ";
	for(auto it = this->layers.begin(); it != this->layers.end(); it++){
		//it->second.print(it->first + " =");
		std::cout<<*it<<" ";
	}
	std::cout<<std::endl;
	BaseModel::get_params();
}

arma::field<arma::mat> MLP::get_coefs(){
	return this->coefs;
}
arma::field<arma::mat> MLP::get_intercepts(){
	return this->intercepts;
}

void MLP::set_params(const std::string& p, const std::vector<int>& layers){
	if(p == "layers"){
		this->layers = layers;
		int n_layers = layers.size();
		this->coefs.set_size(n_layers - 1);
		this->intercepts.set_size(n_layers - 1);
		for(int i = 0; i < n_layers - 1; i++){
			//arma::mat coef(layers[i+1], layers[i]);
			arma::mat coef(layers[i+1], layers[i], arma::fill::randu);
			coefs(i) = coef;
			//arma::mat intercept(layers[i+1], 1);
			//arma::mat intercept(layers[i+1], 1, arma::fill::randu);
			arma::mat intercept(layers[i+1], 1, arma::fill::zeros);
			intercepts(i) = intercept;
		}
		//pkg.unpack(this->coefs, this->intercepts);
	}
	else{
		std::cout<<"set_params failed! did you mean 'layers'?"<<std::endl;
		exit(1);
	}
}


void MLP::fit(const arma::mat& X_train, const arma::mat& y_train){
	arma::mat y = one_hot_encoder(y_train);
	int n_features = X_train.n_cols, n_classes = y.n_cols;
	if (this->fitted == false){
	    this->layers.insert(this->layers.begin(), n_features);
	    this->layers.push_back(n_classes);
	    this->set_params("layers", this->layers);
	    this->fitted = true;
	}
	
	/*
	if(this->layers.front() != n_features || this->layers.back() != n_classes){
		std::vector<int> layers = this->layers;
		layers.front() = n_features;
		layers.back() = n_classes;
		this->set_params("layers", layers);
	}
	*/
	/*
	std::cout<<"get_params in fit method: "<<std::endl;
	this->get_params();
	std::cout<<std::endl;
	return;
	*/
	arma::mat theta = pkg.unpack(this->coefs, this->intercepts);
	if(this->params_string["optimizer"] == "none"){
		SGD solver(this->params_double["learning_rate"]);
		this->_fit_SGD<SGD>(theta, X_train, y, solver);
	}
	if(this->params_string["optimizer"] == "momentum"){
		SGDmomentum solver(this->params_double["learning_rate"],this->params_double["momentum"]);
		this->_fit_SGD<SGDmomentum>(theta, X_train, y, solver);
	}
	arma::field<arma::field<arma::mat>> coefs_inters = pkg.pack(this->params_mat["coef"]);
	this->coefs = coefs_inters(0);
	this->intercepts = coefs_inters(1);
}

arma::mat MLP::predict_proba(const arma::mat& X_test){
	arma::field<arma::field<arma::mat>> ZsAs = this->feed_forward(this->coefs, this->intercepts, X_test);
	arma::field<arma::mat> As = ZsAs(1);
	return As(As.n_elem-1).t();
}

arma::mat MLP::predict(const arma::mat& X_test){
	arma::mat Y_proba = this->predict_proba(X_test);
	arma::umat y_pred_int = arma::index_max(Y_proba, 1);
	arma::mat y_pred = arma::conv_to<arma::mat>::from(y_pred_int);
	return y_pred;
}

double MLP::score(const arma::mat& X, const arma::mat& y){;
	arma::mat y_pred = this->predict(X);
	return accuracy_score(y, y_pred);
}

arma::mat MLP::gradient(const arma::mat& theta, const arma::mat& X, const arma::mat& y){
	arma::field<arma::field<arma::mat>> coefs_inters = pkg.pack(theta);
	//pkg.reset();
	//coefs_inters = pkg.pack(theta);
	arma::field<arma::mat> coefs = coefs_inters(0), intercepts = coefs_inters(1);
	arma::field<arma::field<arma::mat>> grad_coefs_inters = this->back_prop(coefs, intercepts, X, y);
	//grad_coefs_inters = this->back_prop(coefs, intercepts, X, y);
	pkg.reset();
	arma::mat grads = pkg.unpack(grad_coefs_inters(0), grad_coefs_inters(1));
	return grads;
}

//feed_forward will return fields of Z (sum), A (output) at each layer 
arma::field<arma::field<arma::mat>> MLP::feed_forward(const arma::field<arma::mat>& coefs, 
													  const arma::field<arma::mat>& intercepts, 
													  const arma::mat& X){
	int n_ci = coefs.n_elem;//number of (coefs, intercepts) pair
	arma::field<arma::mat> Zs(n_ci), As(n_ci);
	for(int i = 0; i < n_ci; i++){
		if(i == 0){
			arma::mat Z = coefs(i)*X.t();
			Z.each_col() += intercepts(i);
			Zs(i) = Z;
		}
		else{
			arma::mat Z = coefs(i)*As(i-1);
			Z.each_col() += intercepts(i);
			Zs(i) = Z;
		}
		
		if(i == n_ci - 1){//softmax at last layer
			arma::mat A = softmax(Zs(i),0);
			As(i) = A;
		}
		else{
			if(this->params_string["activation"] == "sigmoid"){
				As(i) = sigmoid(Zs(i));
			}
			if(this->params_string["activation"] == "relu"){
				As(i) = relu(Zs(i));
			}
		}
	}
	arma::field<arma::field<arma::mat>> f(2);
	f(0) = Zs;
	f(1) = As;
	return f;
}

arma::field<arma::field<arma::mat>> MLP::back_prop(const arma::field<arma::mat>& coefs, const arma::field<arma::mat>& intercepts,
											  	   const arma::mat& X, const arma::mat& y){
	//coefs, intercepts: field
	//X: n_samples x n_features, y: n_samples x n_classes
	arma::field<arma::field<arma::mat>> coefs_inters;
	coefs_inters = this->feed_forward(coefs, intercepts, X);
	arma::field<arma::mat> Zs = coefs_inters(0), As = coefs_inters(1);
	int n_ci = coefs.n_elem;
	arma::field<arma::mat> Es(n_ci), grad_coefs(n_ci), grad_intercepts(n_ci);
	//Es(n_ci-1) = (1.0/(double(X.n_rows)))*(As(n_ci-1) - y);
	for(int i = n_ci - 1; i >= 0; i--){
		if(i == n_ci-1){
			Es(i) = (1.0/(double(X.n_rows)))*(As(i) - y.t());
		}
		else{
			Es(i) = coefs(i+1).t()*Es(i+1);
			if(this->params_string["activation"] == "sigmoid"){
				Es(i) %= dsigmoid(Zs(i));
			}
			if(this->params_string["activation"] == "relu"){
				Es(i) %= drelu(Zs(i));
			}
		}
		if(i == 0){
			grad_coefs(i) = Es(i)*X;
		}
		else{
			grad_coefs(i) = Es(i)*As(i-1).t();
		}
		grad_intercepts(i) = arma::sum(Es(i), 1);
	}
	arma::field<arma::field<arma::mat>> grads(2);
	grads(0) = grad_coefs;
	grads(1) = grad_intercepts;
	return grads;
}

/*
template <class OPT>
void MLP::_fit_SGD(arma::mat& theta, const arma::mat& X, const arma::mat& y, OPT& solver){
//X: (n_samples x n_features), y: (n_samples x n_classes)
	//arma::uword Xy_n_rows = X.n_rows, X_n_cols = X.n_cols, y_n_cols = y.n_cols;
	
	if(this->params_bool["early_stopping"]){//early_stopping = true
		if(this->params_int.count("n_iter_no_change") == 0){
			this->params_int["n_iter_no_change"] = 10;
		}
		
		arma::field<arma::mat> train_val = train_test_split(X, y, 0.3, 0);//test_size = 0.3, shuffle = 0
		arma::mat X_train, X_val, y_train, y_val;
		X_train = train_val(0); X_val = train_val(1); y_train = train_val(2); y_val = train_val(3);
		
		arma::uword X_train_n_rows = X_train.n_rows, X_train_n_cols = X_train.n_cols, y_train_n_cols = y_train.n_cols;
		std::cout<<"X_train_n_rows: "<<X_train_n_rows<<std::endl;
		std::cout<<"X_train_n_cols: "<<X_train_n_cols<<std::endl;
		std::cout<<"y_train_n_cols: "<<y_train_n_cols<<std::endl;
		X_train.brief_print("X_train:");
		X_val.brief_print("X_val:");
		y_train.brief_print("y_train:");
		y_val.brief_print("y_val:");
		//return;
		double best_score = -1;
		arma::mat best_coef(theta);

		int no_change = 0;
		for(int iter = 0; iter < this->params_int["max_iter"]; iter++){
			if(no_change > this->params_int["n_iter_no_change"]){
				break;
			}
			arma::uvec index = arma::randperm(X_train_n_rows);
			for(arma::uword i = 0; i < X_train_n_rows; i++){
				arma::mat Xi = X_train.submat(index[i],0,index[i],X_train_n_cols-1); 
				arma::mat yi = y_train.submat(index[i],0,index[i],y_train_n_cols-1);
				Xi.brief_print("Xi:");
				yi.print("yi:");
				
				arma::mat grad = this->gradient(theta, Xi, yi);
				solver.update(theta, grad);
			}
			//exit(1);
			//check stopping criteria
			arma::field<arma::field<arma::mat>> ci = pkg.pack(theta);
			//pkg.reset();
			this->coefs = ci(0); this->intercepts = ci(1);
			
			//convert y_val to y (1 column)
			arma::mat y_val_one_col = one_hot_decoder(y_val);
			double curr_score = this->score(X_val, y_val_one_col);
			
			if(curr_score > best_score + this->params_double["tol"]){
				no_change = 0;
				best_score = curr_score;
				best_coef = theta;
			}
			else{
				no_change++;
			}
		}
		arma::field<arma::field<arma::mat>> ci_best = pkg.pack(best_coef);
		this->coefs = ci_best(0); this->intercepts = ci_best(1);
		return;
	}
	
	
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
*/

