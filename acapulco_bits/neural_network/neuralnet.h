#ifndef NEURALNET_H
#define NEURALNET_H

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
class MLP : public BaseModel{
private:
	std::vector<int> layers;
	arma::field<arma::mat> coefs;
	arma::field<arma::mat> intercepts;
	Package pkg;
	bool fitted = false;
public:
	MLP(const std::vector<int>&);
	void init_coefs_inters(arma::field<arma::mat>&, arma::field<arma::mat>&);
	void get_params();
	arma::field<arma::mat> get_coefs();
	arma::field<arma::mat> get_intercepts();
	using BaseModel::set_params;
	void set_params(const std::string&, const std::vector<int>&);
	void fit(const arma::mat&, const arma::mat&);
	arma::mat predict_proba(const arma::mat&);
	arma::mat predict(const arma::mat&);
	double score(const arma::mat&, const arma::mat&);
	arma::mat gradient(const arma::mat&, const arma::mat&, const arma::mat&);
	arma::field<arma::field<arma::mat>> feed_forward(const arma::field<arma::mat>&, const arma::field<arma::mat>&, const arma::mat&);
	arma::field<arma::field<arma::mat>> back_prop(const arma::field<arma::mat>&, const arma::field<arma::mat>&,
												  const arma::mat&, const arma::mat& );
												  
	//override _fit_SGD of BaseModel
	//template <class OPT>
	//void _fit_SGD(arma::mat&, const arma::mat&, const arma::mat&, OPT&);
};

#include "neuralnet.hpp"

#endif
