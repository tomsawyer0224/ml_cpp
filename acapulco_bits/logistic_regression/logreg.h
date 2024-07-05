#ifndef LOGREG_H
#define LOGREG_H

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
class LogisticRegression : public BaseModel{
public:
	void fit(const arma::mat&, const arma::mat&);
	arma::mat predict_proba(const arma::mat&);
	arma::mat predict(const arma::mat&);
	double score(const arma::mat&, const arma::mat&);
	arma::mat gradient(const arma::mat&, const arma::mat&, const arma::mat&);
	//template <class OPT>
	//void _fit(const arma::mat&, const arma::mat&, OPT&);
};

#include "logreg.hpp"

#endif
