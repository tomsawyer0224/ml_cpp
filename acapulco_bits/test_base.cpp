#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include <string>
#include </usr/include/armadillo>
//#include <tuple>
//#include "acapulco.h"

#include "base.h"
#include "sgd.h"

using namespace std;

int main(){
	Base b;
	b.get_params();
	b.set_params("learning_rate", 0.01);
	b.set_params("max_iter", 200);
	b.set_params("optimizer", "none");
	b.set_params("optimizer", "momentum");
	arma::mat A(2,1);
	b.set_params("coef", A);
	b.set_params("verbose", true);
	b.get_params();
	
	cout<<b.get_params_double("learning_rate")<<endl;
	cout<<b.get_params_int("max_iter")<<endl;
	cout<<b.get_params_string("optimizer")<<endl;
	cout<<b.get_params_bool("verbose")<<endl;
	cout<<b.get_params_mat("coef")<<endl;
	   
	return 0;
}
