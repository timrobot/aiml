#ifndef perceptron_train_h
#define perceptron_train_h

#include <armadillo>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <cstdio>
#include "fmincg.h"
#include "highgui.h"
#include "imgproc.h"
#include "dataset.h"

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define print_red(str)			{ printf(ANSI_COLOR_RED			str ANSI_COLOR_RESET); }
#define print_green(str)		{ printf(ANSI_COLOR_GREEN		str ANSI_COLOR_RESET); }
#define print_yellow(str)		{ printf(ANSI_COLOR_YELLOW	str ANSI_COLOR_RESET); }
#define print_blue(str)			{ printf(ANSI_COLOR_BLUE		str ANSI_COLOR_RESET); }
#define print_magenta(str)	{ printf(ANSI_COLOR_MAGENTA	str ANSI_COLOR_RESET); }
#define print_cyan(str)			{ printf(ANSI_COLOR_CYAN		str ANSI_COLOR_RESET); }
#define color_red(str)			{ ANSI_COLOR_RED			str ANSI_COLOR_RESET }
#define color_green(str)		{ ANSI_COLOR_GREEN		str ANSI_COLOR_RESET }
#define color_yellow(str)		{ ANSI_COLOR_YELLOW		str ANSI_COLOR_RESET }
#define color_blue(str)			{ ANSI_COLOR_BLUE			str ANSI_COLOR_RESET }
#define color_magenta(str)	{ ANSI_COLOR_MAGENTA	str ANSI_COLOR_RESET }
#define color_cyan(str)			{ ANSI_COLOR_CYAN			str ANSI_COLOR_RESET }

using namespace arma;

vec normal_eqn(mat X, vec y);
double cost(mat X, vec y, vec theta);
vec grad_descent(mat X, vec y, vec theta, double alpha, int niter, std::vector<double> &costs, double lambda);
double regcost(mat X, vec y, vec theta, double lambda);
vec linreg_train(mat X, vec y, double &err);
mat class_err(mat X, mat y, mat thetas, double &conf, double &err);
mat loadmat(std::string filename);

#endif
