#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include </usr/include/armadillo>
#include <any>
//#include "acapulco.h"

#include "logreg.h"
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
 	arma::mat CHD;
 	CHD.load("framingham_removedNA.csv");
 	//CHD.brief_print("CHD:");
 	arma::uword N = CHD.n_rows;
 	arma::mat X = CHD(arma::span::all, arma::span(0, CHD.n_cols-2));
 	arma::mat y = CHD(arma::span::all, arma::span(CHD.n_cols-1, CHD.n_cols-1));
 	//X.brief_print("X:");
 	//y.brief_print("y:");
	double test_size = 0.3;
	arma::uword N_test = N*test_size;
	arma::uword N_train = N - N_test;
	std::cout<<N_train<<std::endl<<N_test<<std::endl;
	arma::mat X_train = X(arma::span(0, N_train-1),arma::span::all);
	arma::mat y_train = y(arma::span(0, N_train-1),arma::span::all);
	arma::mat X_test = X(arma::span(N_train, N-1),arma::span::all);
	arma::mat y_test = y(arma::span(N_train, N-1),arma::span::all);
	
	LogisticRegression lr_chd;
	lr_chd.set_params("optimizer", "momentum");
	lr_chd.set_params("max_iter", 500);
	lr_chd.set_params("momentum", 0.9);
	lr_chd.get_params();
	 
 	StandardScaler s;
 	
 	arma::mat X_train_scaled = s.fit_transform(X_train);
	arma::mat X_test_scaled = s.transform(X_test);
	lr_chd.fit(X_train_scaled, y_train);
	cout<<"result:"<<endl;
	
	//lr_chd.get_params();
	
	cout<<"n_iters: "<<lr_chd.get_params_int("n_iters")<<endl;
	//lr_chd.get_params();cout<<endl;
	cout<<"accuracy score: "<<lr_chd.score(X_test_scaled, y_test)<<endl;
	
	//y_test.t().save("y_true", arma::csv_ascii);
	//lr_chd.predict(X_test_scaled).t().save("y_pred", arma::csv_ascii);
	/*
	arma::umat y_test_int = arma::conv_to<arma::umat>::from(y_test);
	arma::inplace_trans(y_test_int);
	y_test_int.save("y_true", arma::csv_ascii);
	
	arma::mat y_pred = lr_chd.predict(X_test_scaled);
	arma::umat y_pred_int = arma::conv_to<arma::umat>::from(y_pred);
	arma::inplace_trans(y_pred_int);
	y_pred_int.save("y_pred", arma::csv_ascii);
	*/
	  
	return 0;
}
