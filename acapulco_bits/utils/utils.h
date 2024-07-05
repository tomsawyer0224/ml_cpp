#ifndef UTILS_H
#define UTILS_H

#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include </usr/include/armadillo>
#include <utility>
#include <vector>
#include <cmath>

//std::tuple<arma::mat, arma::mat, arma::mat, arma::mat> train_test_split(const arma::mat&, const arma::mat&, double);
arma::field<arma::mat> train_test_split(const arma::mat&, const arma::mat&, double, bool);
//double sigmoid(const double&);
arma::mat sigmoid(const arma::mat&);
arma::mat dsigmoid(const arma::mat&);

arma::mat softmax(const arma::mat&, int);

arma::mat one_hot_encoder(const arma::mat&, const arma::uword&);
arma::mat one_hot_encoder(const arma::mat&);
arma::mat one_hot_decoder(const arma::mat&);

arma::mat relu(const arma::mat&);
arma::mat drelu(const arma::mat&);

class Package{
private:
//public:
	std::vector<std::pair<arma::uword, arma::uword>> shape1;
	std::vector<std::pair<arma::uword, arma::uword>> shape2;
public:
	arma::mat unpack(const arma::field<arma::mat>&, const arma::field<arma::mat>&);
	arma::field<arma::field<arma::mat>> pack(const arma::mat&);
	void reset();
	void shape();
};

#include "utils.hpp"

#endif
