#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include </usr/include/armadillo>
#include <any>
//#include "acapulco.h"

#include "sofreg.h"
#include "../preprocessing/preprocessing.h"
using namespace std;

int main(){
	/*
 	arma::vec vx = {0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              		2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50};
 	arma::vec vy = {0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1};
 	arma::mat X(vx), y(vy);
	//X.print("X:");
	//y.print("y:");
 	LogisticRegression lr;
 	lr.set_params("max_iter", 10000);
 	lr.set_params("optimizer", "momentum");
 	//lr.set_params("learning_rate", 0.1);
 	lr.get_params();
 	cout<<"-----------------"<<endl;
 	StandardScaler ss;
 	//arma::mat X_train = ss.fit_transform(X);
 	arma::mat X_train = X;
 	lr.fit(X_train, y);
 	arma::mat y_pred = lr.predict(X_train);
 	lr.get_params();
 	y_pred.t().print("predict:");
 	y.t().print("true:");
 	*/
 	/*
 	arma::mat CHD;
 	CHD.load("framingham_removedNA.csv");
 	CHD.brief_print("CHD:");
 	
 	arma::uword N = CHD.n_rows;
 	arma::mat X = CHD(arma::span::all, arma::span(0, CHD.n_cols-2));
 	arma::mat y = CHD(arma::span::all, arma::span(CHD.n_cols-1, CHD.n_cols-1));
 	
 	//X.brief_print("X:");
 	//y.brief_print("y:");
	double test_size = 0.3;
	arma::uword N_test = N*test_size;
	arma::uword N_train = N - N_test;
	//std::cout<<N_train<<std::endl<<N_test<<std::endl;
	arma::mat X_train = X(arma::span(0, N_train-1),arma::span::all);
	arma::mat y_train = y(arma::span(0, N_train-1),arma::span::all);
	arma::mat X_test = X(arma::span(N_train, N-1),arma::span::all);
	arma::mat y_test = y(arma::span(N_train, N-1),arma::span::all);
	
	SoftmaxRegression sr;
	sr.get_params();
	
	StandardScaler ss;
	
	arma::mat X_train_scaled = ss.fit_transform(X_train);
	arma::mat X_test_scaled = ss.transform(X_test);
	sr.fit(X_train_scaled, y_train);
	//sr.get_params();
	arma::mat y_pred_proba = sr.predict_proba(X_test_scaled);
	y_pred_proba.brief_print("y_pred_proba:");
	arma::mat y_pred = sr.predict(X_test);
	cout<<"result:"<<endl;
	cout<<"n_iters: "<<sr.get_params_int("n_iters")<<endl;
	cout<<"accuracy score: "<<sr.score(X_test_scaled, y_test)<<endl;
	*/
	
	arma::mat MP_train;
	MP_train.load("mobile_train.csv");
	MP_train.brief_print("MP_train:");
	
 	arma::mat X = MP_train(arma::span::all, arma::span(0, MP_train.n_cols-2));
 	arma::mat y = MP_train(arma::span::all, arma::span(MP_train.n_cols-1, MP_train.n_cols-1));
 	//X.brief_print("X:");
 	//y.brief_print("y:");
 	
 	double test_size = 0.3;
 	arma::uword N = MP_train.n_rows;
 	arma::uword N_test = N*test_size, N_train = N - N_test;
 	
	arma::mat X_train = X(arma::span(0, N_train-1),arma::span::all);
	arma::mat y_train = y(arma::span(0, N_train-1),arma::span::all);
	arma::mat X_test = X(arma::span(N_train, N-1),arma::span::all);
	arma::mat y_test = y(arma::span(N_train, N-1),arma::span::all);
	//X_train.brief_print("X_train:");
	//y_train.brief_print("y_train:");
	//X_test.brief_print("X_test:");
	//y_test.brief_print("y_test:");
	
	StandardScaler ss;
	arma::mat X_train_scaled = ss.fit_transform(X_train);
	arma::mat X_test_scaled = ss.transform(X_test);
	
	SoftmaxRegression srmp;
	srmp.get_params();
	
	srmp.fit(X_train_scaled, y_train);
	
	arma::mat y_pred_proba = srmp.predict_proba(X_test_scaled);
	y_pred_proba.brief_print("y_pred_proba:");
	arma::mat y_pred = srmp.predict(X_test);
	cout<<"result:"<<endl;
	cout<<"n_iters: "<<srmp.get_params_int("n_iters")<<endl;
	cout<<"accuracy score: "<<srmp.score(X_test_scaled, y_test)<<endl;
	
	return 0;
}
