#ifndef LINREG_H
#define LINREG_H

#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include <cmath>
#include <map>
#include <string>
#include </usr/include/armadillo>

#include "../base.h"
#include "../sgd.h"
#include "../scores/scores.h"
class LinearRegression : public Base{
public:
	void fit(const arma::mat&, const arma::mat&);
	arma::mat predict(const arma::mat&);
	double score(const arma::mat&, const arma::mat&);
};

#include "linreg.hpp"

#endif
