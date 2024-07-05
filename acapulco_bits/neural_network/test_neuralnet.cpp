#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include </usr/include/armadillo>
#include <any>
//#include "acapulco.h"

#include "neuralnet.h"
#include "../preprocessing/preprocessing.h"
using namespace std;

int main(){
	/*
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
	
	MLP mlp_MP({20,5,4});
	//mlp_MP.set_params("n_iter_no_change", 13); 
	//mlp_MP.set_params("max_iter", 10000);
	//mlp_MP.set_params("optimizer", "momentum");
	//mlp_MP.set_params("activation", "relu");
	mlp_MP.get_params();
	cout<<"-----------------"<<endl;
	 
	 
	mlp_MP.fit(X_train_scaled, y_train);
	
	//mlp_MP.get_params();
	//cout<<"-----------------"<<endl;
	arma::mat y_pred_proba = mlp_MP.predict_proba(X_test_scaled);
	y_pred_proba.brief_print("y_pred_proba:");
	arma::mat y_pred = mlp_MP.predict(X_test);
	cout<<"result:"<<endl;
	cout<<"n_iters: "<<mlp_MP.get_params_int("n_iters")<<endl;
	cout<<"accuracy score: "<<mlp_MP.score(X_test_scaled, y_test)<<endl;
	*/
	  
	/*
	MLP m({2,3,4});
	m.get_params();
	cout<<"----------------"<<endl;
	std::vector<int> layers = {4,2,3};
	//m.set_params("layers", std::vector<int>({5,6,7}));
	m.set_params("layers", layers);
	m.set_params("activation", "relu");
	m.set_params("learning_rate", 100.0);
	m.get_params();
	cout<<m.get_params_int("max_iter")<<endl;
	m.get_coefs().print("get_coefs:");
	m.get_intercepts().print("get_intercepts:");
	   
	arma::mat X = {{3,-2,-5,1},{-2,1,2,-1},{-2,4,1,2}};
	arma::mat y = {{0,1,0},{0,0,1},{1,0,0}};
	X.print("X:");
	y.print("y:");
	arma::field<arma::field<arma::mat>> f;
	f = m.feed_forward(m.get_coefs(), m.get_intercepts(), X);
	f(0).print("Zs:");
	f(1).print("As:");
	
	arma::field<arma::field<arma::mat>> grads;
	 
	grads = m.back_prop(m.get_coefs(), m.get_intercepts(), X, y);
	grads(0).print("grad_coefs:");
	grads(1).print("grad_intercepts:");
	*/
	  
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
	/*
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
	*/
	
	
	
	return 0;
}
