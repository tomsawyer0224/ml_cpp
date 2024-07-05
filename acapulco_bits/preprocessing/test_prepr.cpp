#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include </usr/include/armadillo>
//#include "acapulco.h"

#include "preprocessing.h"
using namespace std;
//using namespace aca;

int main(int argc, char** argv){
	arma::vec v1 = arma::linspace(0,29,30);
	v1.print("v1:");
	arma::mat A = reshape(v1,10,3);
	A.print("A:");
	arma::vec v2 = arma::linspace(30,59,30);
	arma::mat B = reshape(v2,10,3);
	B.print("B:");
	
	//StandardScaler ss;
	MinMaxScaler ss;
	arma::mat A_scaled = ss.fit_transform(A);
	arma::mat B_scaled = ss.transform(B);
	A_scaled.print("A_scaled:");
	B_scaled.print("B_scaled:");
	return 0;
}
