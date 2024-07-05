#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include </usr/include/armadillo>
#include <any>
//#include "acapulco.h"

#include "linreg.h"
#include "../preprocessing/preprocessing.h"
using namespace std;

int main(){
	
	LinearRegression lr;
	/*
	lr.set_params("max_iter", 500);
	lr.set_params("optimizer", "none");
	//lr.set_params("optimizer", "momentum");
	lr.set_params("momentum", 0.01);
	lr.get_params();
	*/
	cout<<"-------------"<<endl;
	arma::vec vx = {147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183};
	arma::vec vy = {49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68};
	
	arma::mat X(vx), y(vy); 
	StandardScaler ss;
	//arma::mat X_train = ss.fit_transform(X);
	arma::mat X_train = X;
	//X.print("X:");
	//y.print("y:");
	//X.submat(1,0,1,0).print();
	 
	lr.fit(X_train,y);
	cout<<"--------------\n";
	lr.get_params();
	//lr.get_coef().print("coef:");
	//lr.get_params_mat("coef").print("zzzzzzzzzzz");     
	//cout<<"-------------"<<endl;
	//lr.get_params();

	return 0;
}
