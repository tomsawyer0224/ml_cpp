#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include </usr/include/armadillo>
//#include <tuple>
//#include "acapulco.h"

#include "scores.h"
using namespace std;

int main(){
	/*
	arma::mat m = {{1,2,3,4},{5,6,7,8},{9,10,11,12}};
	m.print("m:");
	arma::mat m1 = arma::shuffle(m);
	m1.print("m1:");
	*/
	arma::vec v1 = arma::linspace(0,29,30);
	v1.print("v1:");
	arma::mat A = reshape(v1,10,3);
	A.print("A:");
	arma::vec v2 = arma::linspace(30,59,30);
	arma::mat B = reshape(v2,10,3);
	B.print("B:");
	
	
	//arma::mat y_true(4,1);
	//y_true(0,0) = 3; y_true(1,0) = -0.5; y_true(2,0) = 2; y_true(3,0) = 7;
	//arma::mat y_pred(4,1);
	//y_pred(0,0) = 2.5; y_pred(1,0) = 0.0; y_pred(2,0) = 2; y_pred(3,0) = 8;
	
	//arma::mat y_true = {{3}, {-0.5}, {2}, {7}};//error when creating arma::mat
	//arma::mat y_pred = {{2.5}, {0.0}, {2}, {8}};//error when creating arma::mat
	//arma::mat y_true = {{3, -0.5, 2, 7}};
	//arma::mat y_pred = {{2.5, 0.0, 2, 8}};
	//arma::vec y_true = {3, -0.5, 2, 7};
	//arma::vec y_pred = {2.5, 0.0, 2, 8};
	//arma::rowvec y_true = {1, 3, 2, 7};
	//arma::rowvec y_pred = {1, 3, 2, 7};
	//cout<<r2_score(y_true, y_pred)<<endl;
	
	
	//arma::mat y_true(4,1);
	//y_true(0,0) = 3; y_true(1,0) = 2; y_true(2,0) = 1; y_true(3,0) = 0;
	//arma::mat y_pred(4,1);
	//y_pred(0,0) = 3; y_pred(1,0) = 2; y_pred(2,0) = 1; y_pred(3,0) = 0.1;
	  
	//arma::mat y_true = {{3, -0.5, 2, 7}};
	//arma::mat y_pred = {{2.5, 0.0, 2, 7.0}};
	//arma::vec y_true = {3, -0.5, 2, 7};
	//arma::vec y_pred = {3, 0.0, 2, 8};
	//arma::rowvec y_true = {1.6, 3, 2, 7};
	//arma::rowvec y_pred = {1, 3, 2, 7};
	//y_true.print("y_true:");
	//y_pred.print("y_pred:");
	//cout<<accuracy_score(y_true, y_pred)<<endl;
	
	//pr(y_pred);
	//pr(y_true);
	
	return 0;
}
