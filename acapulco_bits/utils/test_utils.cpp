#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include </usr/include/armadillo>
#include <tuple>
//#include "acapulco.h"

#include "utils.h"
using namespace std;

int main(){
	/*
	arma::mat Z = {{-1,2,3,-4},{5,6,7,-8}};
	Z.print("Z");

	arma::mat X = sigmoid(Z);
	X.print("sigmoid(Z):");
	
	//cout<<"1 - sigmoid(Z)"<<endl;
	arma::mat XX = 1 - X;
	XX.print("1 - sigmoid(Z):");
	
	arma::mat Y = dsigmoid(Z);
	Y.print("dsigmoid(Z):");
	*/

	//arma::mat X = relu(Z);
	//X.print("relu(Z):");
	//arma::mat Y = drelu(Z);
	//Y.print("drelu(Z):");
	
	/*
	cout<<"--------unpack-----------"<<endl;
	arma::field<arma::mat> f(2);
	arma::mat m1 = {{1,2},{3,4}};
	arma::mat m2 = {{5,6,7,8},{4,5,4,3},{1,1,12,3}};
	f(0) = m1;
	f(1) = m2;
	f.print("f:");

	arma::field<arma::mat> f2(3);
	f2(0) = {{1,2}};
	f2(1) = {{4,5,6},{3,4,2}};
	f2(2) = {{1,2,3,4}};
	
	f2.print("f2");
	
	Package p;
	arma::mat r = p.unpack(f,f2);
	r.print("r:");
	
	cout<<"--------pack-----------"<<endl;
	arma::field<arma::field<arma::mat>> f_pack;
	f_pack = p.pack(r);
	f_pack(0).print("f_pack(0):");
	f_pack(1).print("f_pack(1):");
	cout<<"shape before: "<<endl;p.shape();
	p.reset();
	cout<<"shape after: "<<endl;p.shape();
	*/
	
	/*
	arma::vec v = {0,1,2,2,4,3,2,3,3,4,0};
	arma::uword n_classes = 5;
	arma::mat y(v);
	y.print("y:");
	//arma::mat Y = one_hot_encoder(y, n_classes);
	arma::mat Y = one_hot_encoder(y);
	Y.print("Y:");
	arma::mat y1 = one_hot_decoder(Y);
	y1.print("y1:");
	*/
	/*
	arma::mat Z = {{1,42,5,7},{4,9,70,1},{4,2,0,1}};
	Z = {{1,2,3,4}, {5,6,7,8},{3,1,4,6}};
	Z.print("Z:");
	softmax(Z,1).print("softmax(Z,1)");
	softmax(Z,0).print("softmax(Z,0)");
	*/
	
	/*
	arma::uword r = 3;
	arma::uword c = 2;
	arma::field<arma::mat> F(2,1);
	F(0,0) = arma::mat(r,c, arma::fill::zeros);
	F(1,0) = arma::mat(4,5, arma::fill::ones);
	F.print("F:");
	*/
	/*
	arma::mat Z = {{1,2,3,4},{5,6,7,8}};
	arma::mat S = sigmoid(Z);
	Z.print("Z:");
	S.print("sigmoid(Z):");
	*/
	/*
	arma::mat m = {{1,2,3,4},{5,6,7,8},{9,10,11,12}};
	m.print("m:");
	arma::mat m1 = arma::shuffle(m);
	m1.print("m1:");
	*/
	
	/*
	arma::vec v1 = arma::linspace(0,29,30);
	v1.print("v1:");
	arma::mat A = reshape(v1,10,3);
	A.print("A:");
	arma::vec v2 = arma::linspace(30,59,30);
	arma::mat B = reshape(v2,10,3);
	B.print("B:");
	
	//tuple<arma::mat, arma::mat, arma::mat, arma::mat> res;
	arma::field<arma::mat> res;
	res = train_test_split(A,B, 0.25, false);
	//arma::mat X_train = get<0>(res);
	//arma::mat X_test = get<1>(res);
	//arma::mat y_train = get<2>(res);
	//arma::mat y_test = get<3>(res);
	
	arma::mat X_train = res(0);
	arma::mat X_test = res(1);
	arma::mat y_train = res(2);
	arma::mat y_test = res(3);
	
	X_train.print("X_train:");
	X_test.print("X_test:");
	y_train.print("y_train:");
	y_test.print("y_test:");
	*/
	return 0;
}
