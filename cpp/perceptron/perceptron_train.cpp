#include "perceptron_train.h"

using namespace arma;

vec normal_eqn(mat X, vec y) {
  double m = (double)(int)y.n_rows;
  vec theta = pinv(X.t() * X) * X.t() * y;
  return theta;
}

double cost(mat X, vec y, vec theta) {
  double m = (double)(int)y.n_rows;
  vec diff = X * theta - y;
  double variance = 0.5 / m * dot(diff, diff);
  return variance;
}

vec grad_descent(mat X, vec y, vec theta, double alpha, int niter, std::vector<double> &costs, double lambda) {
  double m = (double)(int)y.n_rows;
  int n = (int)X.n_cols;
  for (int i = 0; i < niter; i++) {
    vec rtheta = join_cols(vec({0}), theta(span(1,n-1)));
    theta -= alpha / m * X.t() * (X * theta - y) +
      alpha / m * lambda * rtheta;
    costs.push_back(cost(X, y, theta));
  }
  return theta;
}

double regcost(mat X, vec y, vec theta, double lambda) {
  double m = (double)(int)y.n_rows;
  int n = (int)X.n_cols;
  vec diff = X * theta - y;
  vec rtheta = join_cols(vec({0}), theta(span(1,n-1)));
  double variance = 0.5 / m * dot(diff, diff) +
    0.5 / m * lambda * dot(rtheta, rtheta);
  return variance;
}

vec linreg_train(mat X, vec y, double &err) {
  vec theta = normal_eqn(X, y);
  err = cost(X, y, theta);
  return theta;
}

void class_err(mat X, vec y, mat thetas, double &conf, double &err) {
  int m = (int)y.n_rows;
  mat fx = join_rows(ones<mat>(m), X) * thetas;
  uvec k(m);
  uvec Y(m);
  for (int i = 0; i < (int)m; i++) {
    double _ = fx.row(i).max(k(i));
    Y(i) = (uword)round(y(i));
		printf(color_cyan("Class: %llu, Prediction: %llu\n"), Y(i), k(i));
		//showimage(_X.row(i).t());
  }
	conf = sum(k == Y) / (double)m;
  err = 1 - conf;
}

mat loadmat(std::string filename) {
  std::ifstream datafile(filename);
  std::string temp;
  getline(datafile, temp);
  int m, n;
  sscanf(temp.c_str(), "[%d,%d]\n", &m, &n);
  mat data(m, n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double d;
      getline(datafile, temp);
      sscanf(temp.c_str(), "%lf\n", &d);
      data(i, j) = d;
    }
  }
  datafile.close();
  return data;
}

void showimage(vec I) {
	mat img = reshape(I, (int)sqrt(I.n_elem), (int)sqrt(I.n_elem));
	img = imresize2(img, 400, 400);
	disp_image("img", img);
	disp_wait();
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("usage: %s train_data train_labels test_data test_labels\n", argv[0]);
    return 0;
  }
	print_green("load the matrices\n");
	mat X, Y;
	int classes;
	opendata(X, Y, classes , argv[1], argv[2]);
  vec y = Y.col(0);
	print_green("Training data...\n");
  X = join_rows(ones<vec>(X.n_rows), X);
  mat thetas(X.n_cols, classes);
  for (int i = 0; i < classes; i++) {
    double _;
    printf(color_red("Training %d\n"), i);
    vec label = (y == ones<vec>(y.n_elem) * i) % ones<vec>(y.n_elem);
    vec theta = linreg_train(X, label, _);
    thetas.col(i) = theta;
  }
	print_green("Training finished!\n");

	print_green("Testing the training...\n");
	opendata(X, Y, classes, argv[3], argv[4]);
	y = Y.col(0);
  double conf, err;
  class_err(X, y, thetas, conf, err);
	print_green("Testing finished!\n");
  printf(color_yellow("Conf: %lf/100, err: %lf/100\n"), conf * 100, err * 100);
  return 0;
}