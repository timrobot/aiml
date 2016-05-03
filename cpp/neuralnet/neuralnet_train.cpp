#include "neuralnet_train.h"

// these now need to be global variables
mat _X, _y;
double _lambda;
vec dimensions;

/** Initialize a bunch of random matrices
 *  @param sizes a vector containing sequential sizes of dimensions. 
 *		for example: vec({ 12, 3, 2 }) will create a 12x3 and 3x2 matrices
 *	@return a vector containing each random matrix that was created
 */
vector<mat> randomWeights(vec sizes) {
	assert(sizes.n_elem >= 2);
	vector<mat> thetas;
	for (int i = 1; i < (int)sizes.n_elem; i++) {
		thetas.push_back(randu<mat>(sizes(i), sizes(i-1)+1));
	}
	return thetas;
}

/** The sigmoid transform
 *  @param z the input signal
 *  @return the output sigmoid signal
 */
mat sigmoid(mat z) {
	return 1.0 / (1.0 + exp(-z));
}

/** The sigmoid gradient transform
 *  @param z the input signal
 *  @return the output sigmoid gradient signal
 */
mat sigmoidGradient(mat z) {
	return sigmoid(z) % (1 - sigmoid(z));
}

/** A cost function specified for the neural network
 *  @param J (output) the cost
 *  @param grad (output) the gradient
 *  @param nn_params the initial theta
 */
void nnCostFunction(double &J, mat &grad, mat &nn_params) {
	mat X = _X;
	mat y = _y;
	double lambda = _lambda;
	int input_layer_size = dimensions(0);
	int hidden_layer_size = dimensions(1);
	int num_labels = dimensions(2);
	int m = (int)X.n_rows;

	// initialise all the theta and theta gradients
	int totalelem1 = (input_layer_size+1) * hidden_layer_size;
	int totalelem2 = (hidden_layer_size+1) * num_labels;
	mat theta1 = reshape(nn_params(span(0, totalelem1-1), span(0,0)),
			hidden_layer_size, input_layer_size + 1);
	mat theta2 = reshape(nn_params(span(totalelem1, totalelem1+totalelem2-1), span(0,0)),
			num_labels, hidden_layer_size + 1);
	mat theta1_grad(theta1.n_rows, theta1.n_cols, fill::zeros);
	mat theta2_grad(theta2.n_rows, theta2.n_cols, fill::zeros);

	// compute the forward propagation matrices
	int K = num_labels - 1; // assume that the labels are 0:1:num_labels-1
	mat Y(m, num_labels, fill::zeros);
	for (int i = 0; i < m; i++) {
		Y(i, y(i)) = 1.0;
	}
	mat A1 = join_rows(ones<mat>(m, 1), X);
	mat Z2 = A1 * theta1.t();
	mat A2 = join_rows(ones<mat>(m, 1), sigmoid(Z2));
	mat Z3 = A2 * theta2.t();
	mat A3 = sigmoid(Z3);

	// compute the cost J
	mat reg1 = theta1.cols(1,theta1.n_cols-1);
	mat reg2 = theta2.cols(1,theta2.n_cols-1);
	J = 1.0 / m * accu((-Y % log(A3) - (1.0 - Y) % log(1.0 - A3))) +
		lambda / (2.0 * m) * (accu(reg1 % reg1) + accu(reg2 % reg2));

	// compute the back propagation matrices
	mat delta_3 = A3 - Y;
	mat A2_grad = join_rows(ones<mat>(m, 1), sigmoidGradient(Z2));
	mat delta_2 = (delta_3 * theta2) % A2_grad;
	theta2_grad = (delta_3.t() * A2) / m;
	theta1_grad = (delta_2.cols(1, delta_2.n_cols-1).t() * A1) / m;

	// compute the regularization for the gradient
	theta2_grad.cols(1, theta2_grad.n_cols-1) += lambda / m * theta2.cols(1, theta2.n_cols-1);
	theta1_grad.cols(1, theta1_grad.n_cols-1) += lambda / m * theta1.cols(1, theta1.n_cols-1);
	grad = join_cols(vectorise(theta1_grad), vectorise(theta2_grad));
}

/** The calling function to train a bunch of thetas given X and y
 *  @param X the input data
 *  @param y the corresponding labels
 *  @param thetas the hypothetical parameters
 *  @param lambda the regularization parameter (recommended = 0.1)
 *  @param max_iter the regularization parameter (recommended = 200)
 *  @return the new trained parameters
 */
vector<mat> nntrain(mat X, mat y, vector<mat> thetas, double lambda, int max_iter) {
	// unroll the parameters
	int input_layer_size = thetas[0].n_cols - 1;
	int hidden_layer_size = thetas[1].n_cols - 1;
	int num_labels = thetas[1].n_rows;
	mat theta1 = vectorise(thetas[0]);
	mat theta2 = vectorise(thetas[1]);
	mat theta = join_cols(theta1, theta2);

	// after unrolling the parameters, then get ready to train using fmincg
	_X = X;
	_y = y;
	_lambda = lambda;
	double cost;
	fmincg2(cost, max_iter, nnCostFunction, theta);

	// reshape the thetas back to the way they were
	int totalelem1 = (input_layer_size + 1) * hidden_layer_size;
	int totalelem2 = (hidden_layer_size + 1) * num_labels;
	theta1 = reshape(theta(span(0, totalelem1-1), span::all),
			hidden_layer_size, input_layer_size + 1);
	theta2 = reshape(theta(span(totalelem1, totalelem1+totalelem2-1), span::all),
			num_labels, hidden_layer_size + 1);
	thetas.clear();
	thetas.push_back(theta1);
	thetas.push_back(theta2);
	return thetas;
}

/** Test the new parameters on the input data
 *  @param X the input data
 *  @param y the labels
 *  @param thetas the new parameters
 *  @param conf (output) the confidence of the parameters
 *  @param err (output) the error of the parameters
 */
void class_err(mat X, mat y, vector<mat> thetas, double &conf, double &err) {
	_X = X; // this is only for displaying the images
	int m = (int)y.n_rows;
	int n = (int)y.n_cols;
	mat theta1 = thetas[0];
	mat theta2 = thetas[1];
	mat a1 = sigmoid(join_rows(ones<vec>(m), X) * theta1.t());
	mat a2 = sigmoid(join_rows(ones<vec>(m), a1) * theta2.t());
	uvec k(m);
	uvec Y(m);
	for (int i = 0; i < m; i++) {
		double _ = a2.row(i).max(k(i));
		Y(i) = (uword)round(y(i,0));
		printf(color_cyan("Class: %llu, Prediction: %llu\n"), Y(i), k(i));
		//showimage(_X.row(i).t());
	}
	conf = sum(k == Y) / (double)m;
	err = 1 - conf;
}

mat loadmat(string filename) {
	ifstream datafile(filename);
	string temp;
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

mat loadcsv(string filename, int n_rows, int n_cols) {
	vector<string> lines;
	char *line = NULL;
	size_t n;
	FILE *fp = fopen(filename.c_str(), "r");
	while (getline(&line, &n, fp) != -1) {
		string s = string(line);
		lines.push_back(line);
		free(line);
		line = NULL;
	}
	mat data(n_rows, n_cols);
	for (int i = 0; i < n_rows; i++) {
		string s = lines[i];
		for (int j = 0; j < n_cols; j++) {
			int found;
			double x;
			if ((found = s.find(",")) != string::npos) {
				x = strtod(s.substr(0, found).c_str(), NULL);
				s = s.substr(found + 1, s.size());
			} else {
				x = strtod(s.substr(0, s.find("\n")).c_str(), NULL);
			}
			data(i, j) = x;
		}
	}
	return data;
}

void showimage(vec I) {
	mat img = reshape(I, (int)sqrt(I.n_elem), (int)sqrt(I.n_elem));
	img = imresize2(img, 400, 400);
	disp_image("img", img);
	disp_wait();
}

int main(int argc, char *argv[]) {
	if (argc != 8) {
		printf("usage: %s train_data train_labels test_data test_labels hidden_layers lambda max_iter\n", argv[0]);
		return 0;
	}
	srand(getpid());
	print_green("load the matrices\n");
	mat X, y;
	int classes;
	opendata(X, y, classes, argv[1], argv[2]);
	double lambda = strtod(argv[6], NULL);
	int input_layer_size = (int)X.n_cols;
	int hidden_layer_size = atoi(argv[5]);

	print_green("Training data...\n");
	vec dims({ (double)input_layer_size, (double)hidden_layer_size, (double)classes });
	vector<mat> thetas = randomWeights(dims);
	dimensions = dims;
	thetas = nntrain(X, y, thetas, lambda, atoi(argv[7]));
	print_green("Training finished!\n");

	print_green("Testing the training...\n");
	opendata(X, y, classes, argv[3], argv[4]);
	double conf, err;
	class_err(X, y, thetas, conf, err);
	print_green("Testing finished!\n");
	printf(color_yellow("Conf: %lf/100, err: %lf/100\n"), conf * 100, err * 100);
	return 0;
}
