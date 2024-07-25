arma::field<arma::mat> train_test_split(const arma::mat& X, const arma::mat& y, double test_size=0.3, bool shuffle=true){
	//return a field: X_train, X_test, y_train, y_test
	if(X.n_rows != y.n_rows){
		std::cout<<"train_test_split() failed! different number of rows"<<std::endl;
		exit(1);
	}
	if(test_size <= 0 || test_size >= 1){
		std::cout<<"train_test_split() failed! test_size must be in (0,1)"<<std::endl;
		exit(1);
	}
	arma::mat Xy = arma::join_rows(X, y);
	if(shuffle){
		Xy = arma::shuffle(Xy);
	}
	arma::uword N = X.n_rows;
	arma::uword N_test = N*test_size;
	if(N_test == N) {
		N_test = N - 1;//ensure that there are at least 1 data point in X_train
	}
	if(N_test == 0){
		N_test = 1;//ensure that there are at least 1 data point in X_test
	}
	arma::uword N_train = N - N_test;
	arma::mat X_train, X_test, y_train, y_test;
	
	X_train = Xy(0, 0, arma::size(N_train, X.n_cols));
	X_test = Xy(N_train, 0, arma::size(N_test, X.n_cols));
	y_train = Xy(0, X.n_cols, arma::size(N_train, y.n_cols));
	y_test = Xy(N_train, X.n_cols, arma::size(N_test, y.n_cols));
	
	arma::field<arma::mat> res(4);
	res(0) = X_train;
	res(1) = X_test;
	res(2) = y_train;
	res(3) = y_test;
	return res;
}


arma::mat sigmoid(const arma::mat& Z){
	return 1.0/(1.0 + arma::exp(-Z));
}

arma::mat dsigmoid(const arma::mat& Z){
	arma::mat res = sigmoid(Z);
	res %= (1.0 - res);
	return res;
}


arma::mat softmax(const arma::mat& Z, int dim = 1){
	//dim = 1, calculate each row
	//dim = 0, calculate each column
	arma::mat maxZ = arma::max(Z, dim);
	arma::mat Z_new(Z), expZ_new, expZ_new_sum;
	if(dim==1){
		//maxZ.print("maxZ:");
		Z_new.each_col() -= maxZ;
		expZ_new = arma::exp(Z_new);
		expZ_new_sum = arma::sum(expZ_new, dim);
		expZ_new.each_col() /= expZ_new_sum;
	}
	else{
		//maxZ.print("maxZ:");
		Z_new.each_row() -= maxZ;
		expZ_new = arma::exp(Z_new);
		expZ_new_sum = arma::sum(expZ_new, dim);
		expZ_new.each_row() /= expZ_new_sum;
	}
	return expZ_new;
}

arma::mat one_hot_encoder(const arma::mat& y, const arma::uword& n_classes){
	//class id must be in range [0, n_classes - 1]
	arma::umat y_int = arma::conv_to<arma::umat>::from(y);
	arma::uword n_samples = y.n_rows;
	arma::mat Y(n_samples, n_classes);
	for(arma::uword i = 0; i < n_samples; i++){
		Y(i, y_int(i, 0)) = 1;
	}
	return Y;
}

arma::mat one_hot_encoder(const arma::mat& y){
	//class id must be in range [0, n_classes - 1]
	arma::umat y_int = arma::conv_to<arma::umat>::from(y);
	arma::uword n_samples = y.n_rows;
	
	arma::uword maximum = y_int.max(), minimum = y_int.min();
	arma::uword n_classes = maximum - minimum + 1;
	
	arma::mat Y(n_samples, n_classes);
	for(arma::uword i = 0; i < n_samples; i++){
		Y(i, y_int(i, 0)) = 1;
	}
	return Y;
}

arma::mat one_hot_decoder(const arma::mat& Y){
	arma::uword n_samples = Y.n_rows, n_classes = Y.n_cols;
	arma::umat index = arma::index_max(Y, 1);
	arma::mat y = arma::conv_to<arma::mat>::from(index);
	return y;
}

//for class Package
arma::mat Package::unpack(const arma::field<arma::mat>& f1, const arma::field<arma::mat>& f2){
	this->reset();
	arma::uword n1 = f1.n_elem, n2 = f2.n_elem;
	arma::mat res(0,1);
	for(arma::uword i = 0; i < n1; i++){
		std::pair<arma::uword, arma::uword> si(f1(i).n_rows, f1(i).n_cols);
		shape1.push_back(si);
		arma::mat fi_flattened = arma::vectorise(f1(i)); 
		res = arma::join_cols(res, fi_flattened);
	}
	for(arma::uword i = 0; i < n2; i++){
		std::pair<arma::uword, arma::uword> si(f2(i).n_rows, f2(i).n_cols);
		shape2.push_back(si);
		arma::mat fi_flattened = arma::vectorise(f2(i)); 
		res = arma::join_cols(res, fi_flattened);
	}
	return res;
}

arma::field<arma::field<arma::mat>> Package::pack(const arma::mat& m){
	arma::uword n1 = this->shape1.size(), n2 = this->shape2.size();
	arma::field<arma::mat> f1(n1), f2(n2);
	arma::uword begin = 0, end = 0;
	for(arma::uword i = 0; i < n1; i++){
		arma::uword n_rows = shape1[i].first, n_cols = shape1[i].second;
		end = begin + n_rows*n_cols;
		arma::mat temp = m.submat(begin, 0, end-1, 0);
		temp.reshape(n_rows, n_cols);
		f1(i) = temp;
		begin = end;
	}
	for(arma::uword i = 0; i < n2; i++){
		arma::uword n_rows = shape2[i].first, n_cols = shape2[i].second;
		end = begin + n_rows*n_cols;
		arma::mat temp = m.submat(begin, 0, end-1, 0);
		temp.reshape(n_rows, n_cols);
		f2(i) = temp;
		begin = end;
	}
	arma::field<arma::field<arma::mat>> f(2);
	f(0) = f1;
	f(1) = f2;
	return f;
}

void Package::reset(){
	this->shape1.clear();
	this->shape2.clear();
}

void Package::shape(){
	std::cout<<"shape1: "<<this->shape1.size()<<std::endl;
	std::cout<<"shape1: "<<this->shape2.size()<<std::endl;
}

//end of class Package

arma::mat relu(const arma::mat& X){
    arma::mat zeros(arma::size(X), arma::fill::zeros);
    arma::mat R = X % (X > zeros);
    return R;
};
arma::mat drelu(const arma::mat& X){
    arma::mat zeros(arma::size(X), arma::fill::zeros);
    arma::mat ones(arma::size(X), arma::fill::ones);
    arma::mat R = ones % (X > zeros);
    return R;
};
