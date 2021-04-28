#pragma once
#include <fstream>
#include <iostream>
#include <string>

#include "ceres/ceres.h"
#include "pgo_error_term.h"
#include "types.h"
#include "read_g2o.h"

using namespace std;

// Constructs the nonlinear least squares optimization problem from the pose
// graph constraints.
void BuildOptimizationProblem(const VectorOfConstraints& constraints,
	MapOfPoses* poses,
	ceres::Problem* problem,
	bool optimize_7dof);

// Returns true if the solve was successful.
bool SolveOptimizationProblem(ceres::Problem* problem, bool debug);

// Output the poses to the file with format: id x y z q_x q_y q_z q_w.
bool OutputPoses(const std::string& filename, const MapOfPoses& poses);