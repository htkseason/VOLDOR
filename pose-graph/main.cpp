#include "pgo.h"


int main(int argc, char** argv) {

	MapOfPoses poses;
	VectorOfConstraints constraints;

	if (argc != 2) {
		cout << "invalid input argument." << endl;
		exit(1);
	}

	if (!ReadG2oFile(string(argv[1]), &poses, &constraints)) {
		cout << "read g2o file failed." << endl;
		exit(1);
	}


	cout << "Number of poses: " << poses.size() << endl;
	cout << "Number of constraints: " << constraints.size() << endl;

	OutputPoses("poses_original.txt", poses);


	if (constraints.size() > 0) {
		ceres::Problem problem;

		BuildOptimizationProblem(constraints, &poses, &problem, true);

		SolveOptimizationProblem(&problem, true);
	}

	OutputPoses("poses_optimized.txt", poses);

	return 0;
}