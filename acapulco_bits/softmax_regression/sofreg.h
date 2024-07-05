#ifndef SOFREG_H
#define SOFREG_H

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
#include "../utils/utils.h"
class SoftmaxRegression : public BaseModel{
public:
	void fit(const arma::mat&, const arma::mat&);
	arma::mat predict_proba(const arma::mat&);
	arma::mat predict(const arma::mat&);
	double score(const arma::mat&, const arma::mat&);
	arma::mat gradient(const arma::mat&, const arma::mat&, const arma::mat&);
};

#include "sofreg.hpp"

#endif
