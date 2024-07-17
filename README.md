# This is a personal project, for educational purposes only!
# About this project:
1. A C/C++ machine learning project that is backed by the Armadillo library.
2. ML models: Linear Regression, Logistic Regression, Softmax Regression, Neural Network.
3. Preprocessing methods: Min Max Scaler, Standard Scaler.
4. Metrics: $R^2$ score, Accuracy score.
5. Optimizers: SGD, SGD momentum.
6. The code is naive and non-optimized.
# How to use:
1. Clone this project.
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

  arma::mat X_train = {...};
  arma::mat X_test = {...};
  arma::mat y_train = {...};
  arma::mat y_test = {...};

  aca::StandardScaler ss;
  arma::mat X_train_scaled = ss.fit_transform(X_train);
  arma::mat X_test_scaled = ss.transform(X_test);
  
  aca::LinearRegression lr;
  lr.fit(X_train_scaled, y_train); //train
  arma::mat y_pred_lr = lr.predict(X_test_scaled); //predict
  double r2 = lr.score(X_test_scaled, y_test); //r2_score
  cout<<"------------------------"<<endl;

  aca::LogisticRegression logreg;
  logreg.fit(X_train_scaled, y_train); //train
  arma::mat y_pred_logreg = logreg.predict(X_test_scaled); //predict
  double acc_logreg = logreg.score(X_test_scaled, y_test); //accuracy score
  cout<<"------------------------"<<endl;

  aca::SoftmaxRegression sr;
  sr.fit(X_train_scaled, y_train); //train
  arma::mat y_pred_sr = sr.predict(X_test_scaled); //predict
  double acc_sr = sr.score(X_test_scaled, y_test); //accuracy score	
  cout<<"------------------------"<<endl;
 
  aca::MLP mlp({3,4,2});
  mlp.fit(X_train_scaled, y_train); //train
  arma::mat y_pred_mlp = mlp.predict(X_test_scaled); //predict
  double acc_mlp = mlp.score(X_test_scaled, y_test); //accuracy score
  
  return 0;
}

```
# Based on:
  https://machinelearningcoban.com \
  https://scikit-learn.org/stable/index.html \
  https://github.com/scikit-learn/scikit-learn
