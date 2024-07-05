MinMaxScaler::MinMaxScaler(){
	min = {};
	max = {};
}

arma::mat MinMaxScaler::fit_transform(const arma::mat& X){
	this->min = arma::min(X, 0);
	this->max = arma::max(X, 0);
	arma::mat X_scaled(X);
	X_scaled.each_row() -= this->min;
	X_scaled.each_row() /= this->max - this->min;
	return X_scaled;
}

arma::mat MinMaxScaler::transform(const arma::mat& X){
	arma::mat X_scaled(X);
	X_scaled.each_row() -= this->min;
	X_scaled.each_row() /= this->max - this->min;
	return X_scaled;
}
