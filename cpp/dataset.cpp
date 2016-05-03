#include <cstdio>
#include "dataset.h"

using namespace arma;
using namespace std;

void opendata(mat &data, mat &labels, int &classes, string datafile, string labelfile) {
	string temp;
	ifstream datafp(datafile);
	if (!datafp.is_open()) {
		printf("Cannot open %s\n", datafile.c_str());
		exit(1);
	}
	vector<string> datastr;
	ifstream labelfp(labelfile);
	if (!labelfp.is_open()) {
		printf("Cannot open %s\n", labelfile.c_str());
		exit(1);
	}
	vector<string> labelstr;
	while (getline(datafp, temp)) {
		datastr.push_back(temp);
	}
	while (getline(labelfp, temp)) {
		labelstr.push_back(temp);
	}
	int datalines = datastr.size();
	int labellines = labelstr.size();
	int imgheight = datalines / labellines;
	int imgwidth = datastr[0].size();

	data = mat(labellines, imgheight * imgwidth);
	labels = mat(labellines, 1);
	for (int l = 0; l < labellines; l++) {
		mat image(imgheight, imgwidth);
		for (int i = 0; i < imgheight; i++) {
			string &line = datastr[l * imgheight + i];
			for (int j = 0; j < imgwidth; j++) {
				image(i, j) = (line[j] == '#') + 0.5 * (line[j] == '+');
			}
		}
		data.row(l) = vectorise(image).t();
		labels(l, 0) = strtod(labelstr[l].substr(0, labelstr[l].find("\n")).c_str(), NULL);
	}
	classes = (int)labels.max() + 1;
}
