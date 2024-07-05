#ifndef SCORES_H
#define SCORES_H

#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include <cmath>
#include </usr/include/armadillo>

double r2_score(const arma::mat&, const arma::mat&);
//double r2_score(const arma::vec&, const arma::vec&);
//double r2_score(const arma::rowvec&, const arma::rowvec&);
template<template<class> class Arma>
double r2_score(const Arma<double>&, const Arma<double>&);

double accuracy_score(const arma::mat&, const arma::mat&);
double accuracy_score(const arma::imat&, const arma::imat&);
//double accuracy_score(const arma::vec&, const arma::vec&);
//double accuracy_score(const arma::rowvec&, const arma::rowvec&);
template<template<class> class Arma>
double accuracy_score(const Arma<double>&, const Arma<double>&);

template <template<class> class U, class T>
void pr(const U<T>&);

#include "scores.hpp"

#endif
