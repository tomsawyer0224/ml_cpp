#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include </usr/include/armadillo>
#include "acapulco.h"

using namespace std;
//using namespace aca;
struct Dataset{
    arma::mat X_train;
    arma::mat y_train;
    arma::mat X_test;
    arma::mat y_test;
};
Dataset get_dataset(string path, string test_path = "", double test_size = 0.3, bool normalize = true){
    arma::mat X_train, y_train, X_test, y_test;
    if (test_path == ""){
        arma::mat data;
        data.load(path);
     	arma::uword N = data.n_rows;
     	arma::mat X = data(arma::span::all, arma::span(0, data.n_cols-2));
     	arma::mat y = data(arma::span::all, arma::span(data.n_cols-1, data.n_cols-1));

	    arma::uword N_test = N*test_size;
	    arma::uword N_train = N - N_test;
	    X_train = X(arma::span(0, N_train-1), arma::span::all);
	    y_train = y(arma::span(0, N_train-1), arma::span::all);
	    X_test = X(arma::span(N_train, N-1), arma::span::all);
	    y_test = y(arma::span(N_train, N-1), arma::span::all);
    } else {
        arma::mat train_data, test_data;
        train_data.load(path);
        test_data.load(test_path);
        X_train = train_data(arma::span::all, arma::span(0, train_data.n_cols-2));
        y_train = train_data(arma::span::all, arma::span(train_data.n_cols-1, train_data.n_cols-1));
        X_test = test_data(arma::span::all, arma::span(0, test_data.n_cols-2));
        y_test = test_data(arma::span::all, arma::span(test_data.n_cols-1, test_data.n_cols-1));
    }
    if (normalize){
        aca::StandardScaler scaler;
        X_train = scaler.fit_transform(X_train);
        X_test = scaler.transform(X_test);
    }
    Dataset ds = {
        X_train,
        y_train,
        X_test,
        y_test,
    };
    return ds;
};
void print_dataset(Dataset ds){
    ds.X_train.brief_print("X_train:");
    ds.y_train.brief_print("y_train:");
    ds.X_test.brief_print("X_test:");
    ds.y_test.brief_print("y_test:");
}
int main(int argc, char** argv){
    Dataset advertising_ds = get_dataset("./datasets/Advertising_no_header.csv");
    Dataset framingham_ds = get_dataset("./datasets/framingham_removedNA_no_header.csv");
    Dataset mobile_ds = get_dataset("./datasets/mobile_train_no_header.csv");
    
    /*
    cout<<"advertising_ds"<<endl;
    print_dataset(advertising_ds);
    cout<<"------------------------"<<endl;
    
    cout<<"framingham_ds"<<endl;
    print_dataset(framingham_ds);
    cout<<"------------------------"<<endl;
    
    cout<<"mobile_ds"<<endl;
    print_dataset(mobile_ds);
    cout<<"------------------------"<<endl;
    */
    
    cout<<"-------------------------------------"<<endl;
    // Linear Regression
    cout<<"Linear Regression on Advertising dataset"<<endl;
    aca::LinearRegression lin_reg;
    lin_reg.fit(advertising_ds.X_train, advertising_ds.y_train);
    double r2_train = lin_reg.score(advertising_ds.X_train, advertising_ds.y_train);
    double r2 = lin_reg.score(advertising_ds.X_test, advertising_ds.y_test);
    cout<<"Training:"<<endl;
    cout<<"     number of training samples: "<<advertising_ds.X_train.n_rows<<endl;
    cout<<"     R2: "<<r2_train<<endl;
    cout<<"Testing:"<<endl;
    cout<<"     number of testing samples: "<<advertising_ds.X_test.n_rows<<endl;
    cout<<"     R2: "<<r2<<endl;
    cout<<"-------------------------------------"<<endl;
    
    // Logistic Regression
    cout<<"Logistic Regression on framingham dataset"<<endl;
    aca::LogisticRegression log_reg;
    log_reg.fit(framingham_ds.X_train, framingham_ds.y_train);
    double acc_log_reg_train = log_reg.score(framingham_ds.X_train, framingham_ds.y_train);
    double acc_log_reg = log_reg.score(framingham_ds.X_test, framingham_ds.y_test);
    cout<<"Training:"<<endl;
    cout<<"     number of training samples: "<<framingham_ds.X_train.n_rows<<endl;
    cout<<"     number of iterations: "<<log_reg.get_params_int("n_iters")<<endl;
    cout<<"     accuracy: "<<acc_log_reg_train<<endl;
    cout<<"Testing:"<<endl;
    cout<<"     number of testing samples: "<<framingham_ds.X_test.n_rows<<endl;
    cout<<"     accuracy: "<<acc_log_reg<<endl;
    cout<<"-------------------------------------"<<endl;
    
    // Softmax Regression
    cout<<"Softmax Regression on mobile dataset"<<endl;
    aca::SoftmaxRegression sof_reg;
    sof_reg.fit(mobile_ds.X_train, mobile_ds.y_train);
    double acc_soft_reg_train = sof_reg.score(mobile_ds.X_train, mobile_ds.y_train);
    double acc_soft_reg = sof_reg.score(mobile_ds.X_test, mobile_ds.y_test);
    cout<<"Training:"<<endl;
    cout<<"     number of training samples: "<<mobile_ds.X_train.n_rows<<endl;
    cout<<"     number of iterations: "<<sof_reg.get_params_int("n_iters")<<endl;
    cout<<"     accuracy: "<<acc_soft_reg_train<<endl;
    cout<<"Testing:"<<endl;
    cout<<"     number of testing samples: "<<mobile_ds.X_test.n_rows<<endl;
    cout<<"     accuracy: "<<acc_soft_reg<<endl;
    cout<<"-------------------------------------"<<endl;
    
    // Neural Network
    cout<<"Neural Network  on mobile dataset"<<endl;
    aca::MLP neural_network({10}); // hidden layers; n_features = 20, n_classes = 4 -> {20, 10, 4}
    //neural_network.set_params("max_iter", 300);
    neural_network.set_params("activation", "relu");
    neural_network.set_params("optimizer", "momentum");
    
    neural_network.fit(mobile_ds.X_train, mobile_ds.y_train);

    double acc_nn_train = neural_network.score(mobile_ds.X_train, mobile_ds.y_train);
    double acc_nn = neural_network.score(mobile_ds.X_test, mobile_ds.y_test);
    cout<<"Training:"<<endl;
    cout<<"     number of training samples: "<<mobile_ds.X_train.n_rows<<endl;
    cout<<"     number of iterations: "<<neural_network.get_params_int("n_iters")<<endl;
    cout<<"     accuracy: "<<acc_nn_train<<endl;
    cout<<"Testing:"<<endl;
    cout<<"     number of testing samples: "<<mobile_ds.X_test.n_rows<<endl;
    cout<<"     accuracy: "<<acc_nn<<endl;
    cout<<"-------------------------------------"<<endl;

    return 0;
}
/*
-------------------------------------
Linear Regression on Advertising dataset
Training:
     number of training samples: 140
     R2: 0.89371
Testing:
     number of testing samples: 60
     R2: 0.901074
-------------------------------------
Logistic Regression on framingham dataset
Training:
     number of training samples: 2560
     number of iterations: 200
     accuracy: 0.853906
Testing:
     number of testing samples: 1096
     accuracy: 0.849453
-------------------------------------
Softmax Regression on mobile dataset
Training:
     number of training samples: 1400
     number of iterations: 200
     accuracy: 0.98
Testing:
     number of testing samples: 600
     accuracy: 0.965
-------------------------------------
Neural Network  on mobile dataset
Training:
     number of training samples: 1400
     number of iterations: 200
     accuracy: 0.950714
Testing:
     number of testing samples: 600
     accuracy: 0.913333
-------------------------------------
*/

