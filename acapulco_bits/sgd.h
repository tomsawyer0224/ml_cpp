#ifndef SGD_H
#define SGD_H

#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include <cmath>
#include <map>
#include <string>
#include </usr/include/armadillo>

class SGD : public BaseOptimizer{
public:
	SGD(double);
	arma::mat get_update(const arma::mat&);
};


class SGDmomentum : public BaseOptimizer{
public:
	SGDmomentum(double, double);
	arma::mat get_update(const arma::mat&);
};

#include "sgd.hpp"
#endif
