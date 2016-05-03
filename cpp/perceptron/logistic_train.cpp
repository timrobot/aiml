#include "logistic_train.h"

// these now need to be global variables
mat _X, _y;
double _lambda;

mat sigmoid(mat z) {
	return 1.0 / (1.0 + exp(-z));
}

void lrCostFunction(double &J, mat &grad, mat &_theta) {
	mat X = _X;
	mat y = _y;
	double lambda = _lambda;
	vec theta = _theta.col(0);
	int m = (int)X.n_rows;
	int n = (int)X.n_cols;
	X = join_rows(ones<vec>(m), X);
	vec reg = join_cols(zeros<vec>(1), theta(span(1,n)));
	mat variance = 1.0 / m * (-y.t() * log(sigmoid(X * theta)) -
			(1.0 - y).t() * log(1.0 - sigmoid(X * theta))) +
			lambda / (2.0 * m) * (reg.t() * reg);

	J = variance(0, 0);
	grad = 1.0 / m * X.t() * (sigmoid(X * theta) - y) +
			lambda / m * reg;
	grad = vectorise(grad);
}

mat oneVsAll(mat X, mat y, int classes, double lambda) {
	int m = (int)X.n_rows;
	int n = (int)X.n_cols;
	mat theta(classes, n + 1, fill::zeros);
	for (int i = 0; i < classes; i++) {
		printf(color_red("Training %d\n"), i);
		mat t(n + 1, 1, fill::zeros);
		double cost;
		_X = X;
		_y = (y == i) % ones<mat>(m, 1);
		_lambda = lambda;
		fmincg2(cost, 50, lrCostFunction, t);
		theta(span(i,i), span::all) = t.t();
	}
	return theta;
}

void class_err(mat X, mat y, mat thetas, double &conf, double &err) {
	int m = (int)y.n_rows;
	int n = (int)y.n_cols;
	X = join_rows(ones<vec>(m), X);
	mat fx = sigmoid(X * thetas);
	uvec k(m);
	uvec Y(m);
	for (int i = 0; i < m; i++) {
		double _ = fx.row(i).max(k(i));
		Y(i) = (uword)round(y(i,0));
		//printf(color_cyan("Class: %llu, Prediction: %llu\n"), Y(i), k(i));
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
	if (argc != 3) {
		printf("usage: %s data labels\n", argv[0]);
		return 0;
	}
	int classes = 10;
	// load the matrices
	mat X = loadmat(argv[1]);
	mat y = loadmat(argv[2]);
	double lambda = 0.1;

	print_green("Training data...\n");
	mat thetas = oneVsAll(X, y, classes, lambda);
	print_green("Training finished!\n");

	print_green("Testing the training...\n");
	double conf, err;
	class_err(X, y, thetas.t(), conf, err);
	print_green("Testing finished!\n");
	printf(color_yellow("Conf: %lf/100, err: %lf/100\n"), conf * 100, err * 100);
	return 0;
}
