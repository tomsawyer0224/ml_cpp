#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include </usr/include/armadillo>

class StandardScaler{
private:
	arma::rowvec mean;//mean of each features
	arma::rowvec sd;//standard deviation of each features
public:
	StandardScaler();
	arma::mat fit_transform(const arma::mat&);
	arma::mat transform(const arma::mat&);
	//arma::rowvec get_mean();
	//arma::rowvec get_sd();
};
#include "ss.hpp"


class MinMaxScaler{
private:
	arma::rowvec min;
	arma::rowvec max;
public:
	MinMaxScaler();
	arma::mat fit_transform(const arma::mat&);
	arma::mat transform(const arma::mat&);
};
#include "mm.hpp"

#endif
