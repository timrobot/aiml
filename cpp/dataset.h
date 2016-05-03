#ifndef dataset_h
#define dataset_h

#include <armadillo>
#include <vector>
#include <string>

void opendata(arma::mat &data, arma::mat &labels, int &classes, std::string datafile, std::string labelfile);

#endif
