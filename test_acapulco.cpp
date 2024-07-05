#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include </usr/include/armadillo>
#include "acapulco.h"


using namespace std;
//using namespace aca;

int main(int argc, char** argv){

	aca::LinearRegression lr;
	lr.get_params();
	cout<<"------------------------"<<endl;

	aca::LogisticRegression logreg;
	logreg.get_params();
	cout<<"------------------------"<<endl;

	aca::SoftmaxRegression sr;
	sr.get_params();	
	cout<<"------------------------"<<endl;
	aca::MLP mlp({3,4,2});
	mlp.get_params();
	return 0;
}
