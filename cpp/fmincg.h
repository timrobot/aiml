#ifndef fmincg_h
#define fmincg_h

#include <armadillo>

typedef void (*CostFunction)(double &J, arma::mat &grad, arma::mat &theta);

void fmincg2(double& finalcost, const int length, CostFunction costfunction, arma::mat& nn_params);//, const int input_layer_size,const int hidden_layer_size,const int num_labels,mat& inputdata,mat& y,const double lambda);

#endif
