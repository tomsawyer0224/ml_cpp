#ifndef ACAPULCO_H
#define ACAPULCO_H

#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <iostream>
#include </usr/include/armadillo>

namespace aca{
	#include "acapulco_bits/preprocessing/preprocessing.h"
	#include "acapulco_bits/utils/utils.h"
	#include "acapulco_bits/scores/scores.h"
	#include "acapulco_bits/linear_regression/linreg.h"
	#include "acapulco_bits/logistic_regression/logreg.h"
	#include "acapulco_bits/softmax_regression/sofreg.h"
	#include "acapulco_bits/neural_network/neuralnet.h"
}

#endif
