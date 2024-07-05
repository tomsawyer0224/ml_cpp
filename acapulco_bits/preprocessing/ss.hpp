StandardScaler::StandardScaler(){
	mean = {};
	sd = {};
}

arma::mat StandardScaler::fit_transform(const arma::mat& X){
	this->mean = arma::mean(X, 0);
	this->sd = arma::stddev(X, 1, 0);
	arma::mat X_scaled(X);
	X_scaled.each_row() -= mean;
	X_scaled.each_row() /= sd;
	return X_scaled;
}

arma::mat StandardScaler::transform(const arma::mat& X){
	arma::mat X_scaled(X);
	X_scaled.each_row() -= mean;
	X_scaled.each_row() /= sd;
	return X_scaled;
}
