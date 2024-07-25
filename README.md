# Machine learning algorithms in C/C++
This is a C/C++ machine learning project that is backed by the Armadillo library.
# About this project
- This is a personal project, for educational purposes only!
- This project was built to help understand the core concepts of machine learning algorithms and optimization algorithms: loss function, gradient descent, etc.
- Components:
    - Machine learning models: Linear Regression, Logistic Regression, Softmax Regression, Neural Network.
    - Preprocessing methods: Min Max Scaler, Standard Scaler.
    - Metrics: $R^2$ score, accuracy score.
    - Optimizers: SGD, SGD momentum.
- The code is naive and non-optimized.
# Experiment
1. **Training**
    - Experiment 1: Predict sales based on advertising expenditures (on TV, radio, and newspaper).
        - Model: Linear Regression
        - Dataset: [Advertising](datasets/Advertising.csv)
    - Experiment 2: Predict whether the patient has a 10-year risk of future (CHD) coronary heart disease.
        - Model: Logistic Regression
        - Dataset: [framingham](datasets/framingham_removedNA_org.csv)
    - Experiment 3: Predict the price range of a mobile phone.
        - Models: Softmax Regression, Neural Network
        - Dataset: [mobile](datasets/mobile_train_org.csv)
    - Platform: Ubuntu 22.04.4 LTS (RAM 12GB, Processor Core i5-3470).
2. **Results**
    ```
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
    ```
4. **Conclusions**
# How to use
1. Clone this repo and cd into ml_cpp.
2. Install the requirements: gcc compiler, armadilo library, CMake.
3. Example:
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
