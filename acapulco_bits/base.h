#ifndef BASE_H
#define BASE_H

#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
//#include <cmath>
#include <map>
#include <string>
#include <cstdbool>
#include </usr/include/armadillo>

class Base{
//base class for all class
protected:
	std::map<std::string, double> params_double;//stores double parameters
	std::map<std::string, std::string> params_string;//stores string parameters
	std::map<std::string, int> params_int;//stores int parameters
	std::map<std::string, bool> params_bool;//stores bool parameters
	std::map<std::string, arma::mat> params_mat;//stores arma::mat parameters(coefs, intercepts)
	
public:
	void get_params();//show all parameters
	//return value of specific parameter
	double get_params_double(const std::string&);
	std::string get_params_string(const std::string&);
	int get_params_int(const std::string&);
	bool get_params_bool(const std::string&);
	arma::mat get_params_mat(const std::string&);
	
	void set_params(const std::string&, const double&);
	void set_params(const std::string&, const char*);//const std::string is ambiguous with arma::mat
	//because arma::mat have a constructor with std::string
	void set_params(const std::string&, const int&);
	void set_params(const std::string&, const bool&);
	void set_params(const std::string&, const arma::mat&);
};

class BaseOptimizer : public Base{
//base class for all optimizers (gradient-based)
public:
	BaseOptimizer(double);
	void update(arma::mat&, const arma::mat&);
	virtual arma::mat get_update(const arma::mat&) = 0;
};

class BaseModel : public Base{
public:
	BaseModel();
	virtual void fit(const arma::mat&, const arma::mat&) = 0;//X_train, y_train
	virtual arma::mat predict(const arma::mat&) = 0;//X_test
	virtual double score(const arma::mat&, const arma::mat&) = 0;//X_test, y_test
	virtual arma::mat gradient(const arma::mat&, const arma::mat&, const arma::mat&) = 0;
	//template <class OPT>
	//void _fit_SGD(const arma::mat&, const arma::mat&, OPT&);
	template <class OPT>
	void _fit_SGD(arma::mat&, const arma::mat&, const arma::mat&, OPT&);
};

#include "base.hpp"
#endif
