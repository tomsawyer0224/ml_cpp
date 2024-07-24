# Machine learning algorithms in C/C++
This is a C/C++ machine learning project that is backed by the Armadillo library.
# About this project
- This is a personal project, for educational purposes only!
- This project was built to help understand the core concepts of machine learning algorithms and optimization algorithms: the loss function, the gradient descent method, etc.
- Components:
    - Machine learning models: Linear Regression, Logistic Regression, Softmax Regression, Neural Network.
    - Preprocessing methods: Min Max Scaler, Standard Scaler.
    - Metrics: $R^2$ score, accuracy score.
    - Optimizers: SGD, SGD momentum.
- The code is naive and non-optimized.
# How to use
1. Clone this project.
2. Install the requirements: gcc compiler, armadilo library, and CMake.
3. Example
    ```
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
        arma::uword N = CHD.n_rows;
        arma::mat X = CHD(arma::span::all, arma::span(0, CHD.n_cols-2));
        arma::mat y = CHD(arma::span::all, arma::span(CHD.n_cols-1, CHD.n_cols-1));
        double test_size = 0.3;
        arma::uword N_test = N*test_size;
        arma::uword N_train = N - N_test;
        arma::mat X_train = X(arma::span(0, N_train-1),arma::span::all);
        arma::mat y_train = y(arma::span(0, N_train-1),arma::span::all);
        arma::mat X_test = X(arma::span(N_train, N-1),arma::span::all);
        arma::mat y_test = y(arma::span(N_train, N-1),arma::span::all);
        
        aca::LogisticRegression lr_chd;
        lr_chd.set_params("optimizer", "momentum");
        lr_chd.set_params("max_iter", 500);
        lr_chd.set_params("momentum", 0.9);
    
        aca::StandardScaler s;
        
        arma::mat X_train_scaled = s.fit_transform(X_train);
        arma::mat X_test_scaled = s.transform(X_test);
        lr_chd.fit(X_train_scaled, y_train);
        cout<<"Training phase:"<<endl;
        cout<<"     number of training samples: "<<N_train<<endl;
        cout<<"     number of iterations: "<<lr_chd.get_params_int("n_iters")<<endl;
        cout<<"Testing phase:"<<endl;
        cout<<"     number of testing samples: "<<N_test<<endl;
        cout<<"     accuracy score: "<<lr_chd.score(X_test_scaled, y_test)<<endl;
    
        return 0;
    }
    ```
    ```
    Result:
    Training phase:
         number of training samples: 2560
         number of iterations: 500
    Testing phase:
         number of testing samples: 1096
         accuracy score: 0.834854
    ```
# Based on:
  https://machinelearningcoban.com \
  https://scikit-learn.org/stable/index.html \
  https://github.com/scikit-learn/scikit-learn
