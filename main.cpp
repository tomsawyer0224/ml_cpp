#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include </usr/include/armadillo>
#include "acapulco.h"

using namespace std;
//using namespace aca;

int main(int argc, char** argv){
 	arma::mat CHD;
 	CHD.load("./datasets/framingham_removedNA.csv");
 	//CHD.brief_print("CHD:");
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
	
	aca::LogisticRegression lr_chd;
	lr_chd.set_params("optimizer", "momentum");
	lr_chd.set_params("max_iter", 500);
	lr_chd.set_params("momentum", 0.9);
	//lr_chd.get_params();
	 
 	aca::StandardScaler s;
 	
 	arma::mat X_train_scaled = s.fit_transform(X_train);
	arma::mat X_test_scaled = s.transform(X_test);
	lr_chd.fit(X_train_scaled, y_train);
	//cout<<"result:"<<endl;
	
	//lr_chd.get_params();
	cout<<"Training phase:"<<endl;
    cout<<"     number of training samples: "<<N_train<<endl;
	cout<<"     n_iters: "<<lr_chd.get_params_int("n_iters")<<endl;
	//lr_chd.get_params();cout<<endl;
    cout<<"Testing phase:"<<endl;
    cout<<"     number of testing samples: "<<N_test<<endl;
	cout<<"     accuracy score: "<<lr_chd.score(X_test_scaled, y_test)<<endl;
    
    /*
    arma::mat X_train = {10.0, 11.2, 30.4};
    arma::mat X_test = {2.4, 7.9};
    arma::mat y_train = {2.0, 3.5, 5.6};
    arma::mat y_test = {1.2, 2.3};

    aca::StandardScaler ss;
    arma::mat X_train_scaled = ss.fit_transform(X_train);
    arma::mat X_test_scaled = ss.transform(X_test);

    cout<<"Linear Regression"<<endl;    
    aca::LinearRegression lr;
    lr.fit(X_train_scaled, y_train); //train
    arma::mat y_pred_lr = lr.predict(X_test_scaled); //predict
    double r2 = lr.score(X_test_scaled, y_test); //r2_score
    cout<<"R2 score: "<<r2<<endl;
    cout<<"------------------------"<<endl;

    cout<<"Logistic Regression"<<endl; 
    aca::LogisticRegression logreg;
    logreg.fit(X_train_scaled, y_train); //train
    arma::mat y_pred_logreg = logreg.predict(X_test_scaled); //predict
    double acc_logreg = logreg.score(X_test_scaled, y_test); //accuracy score
    cout<<"accuracy: "<<acc_logreg<<endl;
    cout<<"------------------------"<<endl;

    cout<<"Softmax Regression"<<endl; 
    aca::SoftmaxRegression sr;
    sr.fit(X_train_scaled, y_train); //train
    arma::mat y_pred_sr = sr.predict(X_test_scaled); //predict
    double acc_sr = sr.score(X_test_scaled, y_test); //accuracy score	
    cout<<"accuracy: "<<acc_sr<<endl;
    cout<<"------------------------"<<endl;

    cout<<"Neural Network"<<endl; 
    aca::MLP mlp({3,4,2});
    mlp.fit(X_train_scaled, y_train); //train
    arma::mat y_pred_mlp = mlp.predict(X_test_scaled); //predict
    double acc_mlp = mlp.score(X_test_scaled, y_test); //accuracy score
    cout<<"accuracy: "<<acc_mlp<<endl;
    cout<<"------------------------"<<endl;
    */
    

    return 0;
}

