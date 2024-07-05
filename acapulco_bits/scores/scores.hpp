
double r2_score(const arma::mat& y_true, const arma::mat& y_pred){
	if(y_true.n_rows != y_pred.n_rows || y_true.n_cols != y_pred.n_cols){
		std::cout<<"r2_score() failed! different size"<<std::endl;
		exit(1);
	}
	arma::mat residual = y_true - y_pred;
	int dim = 0;
	if(y_true.n_rows == 1){
		dim = 1;
	}
	double mean = arma::as_scalar(arma::mean(y_true, dim));
	arma::mat total = y_true - mean;
	double SSres;
	double SStot;
	SSres = arma::norm(residual, "fro")*arma::norm(residual, "fro");
	SStot = arma::norm(total, "fro")*arma::norm(total, "fro");
	return 1.0 - SSres/SStot; 
}

template<template<class> class Arma>
double r2_score(const Arma<double>& y_true, const Arma<double>& y_pred){
	if(y_true.n_elem != y_pred.n_elem){
		std::cout<<"r2_score() failed! different size"<<std::endl;
		exit(1);
	}
	Arma<double> residual = y_true - y_pred;
	Arma<double> total = y_true - arma::mean(y_true);
	double SSres = arma::norm(residual)*arma::norm(residual);
	double SStot = arma::norm(total)*arma::norm(total);
	return 1.0 - SSres/SStot;

}



double accuracy_score(const arma::mat& y_true, const arma::mat& y_pred){
	if(y_true.n_rows != y_pred.n_rows || y_true.n_cols != y_pred.n_cols){
		std::cout<<"accuracy_score() failed! different size"<<std::endl;
		exit(1);
	}
	arma::uword count_same = 0;
	for(arma::uword i = 0; i < y_true.n_elem; i++){
		if(fabs(y_true[i] - y_pred[i]) < 0.1){
			count_same++;
		}
	}
	return double(count_same)/double(y_true.n_elem);
}

template<template<class> class Arma>
double accuracy_score(const Arma<double>& y_true, const Arma<double>& y_pred){
	if(y_true.n_elem != y_pred.n_elem){
		std::cout<<"accuracy_score() failed! different size"<<std::endl;
		exit(1);
	}
	arma::uword count_same = 0;
	for(arma::uword i = 0; i < y_true.n_elem; i++){
		if(fabs(y_true[i] - y_pred[i]) < 0.1){
			count_same++;
		}
	}
	return double(count_same)/double(y_true.n_elem);
}


template <template<class> class U, class T>
void pr(const U<T>& x){
	x.print();
}

