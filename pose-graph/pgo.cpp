#include "pgo.h"

using namespace std;


// Constructs the nonlinear least squares optimization problem from the pose
// graph constraints.
void BuildOptimizationProblem(const VectorOfConstraints& constraints,
	MapOfPoses* poses,
	ceres::Problem* problem,
	bool optimize_7dof) {

	ceres::LossFunction* loss_function = NULL;
	ceres::LocalParameterization* quaternion_local_parameterization =
		new ceres::EigenQuaternionParameterization;

	for (VectorOfConstraints::const_iterator constraints_iter =
		constraints.begin();
		constraints_iter != constraints.end();
		++constraints_iter) {
		const Constraint3d& constraint = *constraints_iter;

		MapOfPoses::iterator pose_begin_iter = poses->find(constraint.id_begin);
		if (pose_begin_iter == poses->end()) {
			cout << "Pose with ID: " << constraint.id_begin << " not found." << endl;
			exit(1);
		}
		MapOfPoses::iterator pose_end_iter = poses->find(constraint.id_end);
		if (pose_end_iter == poses->end()) {
			cout << "Pose with ID: " << constraint.id_end << " not found." << endl;
			exit(1);
		}

		const Eigen::Matrix<double, 7, 7> sqrt_information =
			constraint.information.llt().matrixL();

		// Ceres will take ownership of the pointer.
		ceres::CostFunction* cost_function =
			PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);

		// NOTE that I swapped end/begin to make the constraint pose describes begin->end
		problem->AddResidualBlock(cost_function,
			loss_function,
			pose_end_iter->second.p.data(),
			pose_end_iter->second.q.coeffs().data(),
			&pose_end_iter->second.s,
			pose_begin_iter->second.p.data(),
			pose_begin_iter->second.q.coeffs().data(),
			&pose_begin_iter->second.s);

		problem->SetParameterization(pose_begin_iter->second.q.coeffs().data(),
			quaternion_local_parameterization);
		problem->SetParameterization(pose_end_iter->second.q.coeffs().data(),
			quaternion_local_parameterization);

		if (!optimize_7dof) {
			problem->SetParameterBlockConstant(&pose_begin_iter->second.s);
			problem->SetParameterBlockConstant(&pose_end_iter->second.s);
		}
	}


	// The pose graph optimization problem has six DOFs that are not fully
	// constrained. This is typically referred to as gauge freedom. You can apply
	// a rigid body transformation to all the nodes and the optimization problem
	// will still have the exact same cost. The Levenberg-Marquardt algorithm has
	// internal damping which mitigates this issue, but it is better to properly
	// constrain the gauge freedom. This can be done by setting one of the poses
	// as constant so the optimizer cannot change it.
	MapOfPoses::iterator pose_start_iter = poses->begin();

	problem->SetParameterBlockConstant(pose_start_iter->second.p.data());
	problem->SetParameterBlockConstant(pose_start_iter->second.q.coeffs().data());
	problem->SetParameterBlockConstant(&pose_start_iter->second.s);

	


}

// Returns true if the solve was successful.
bool SolveOptimizationProblem(ceres::Problem* problem, bool debug) {

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

	ceres::Solver::Summary summary;
	ceres::Solve(options, problem, &summary);

	if (debug)
		std::cout << summary.FullReport() << '\n';

	return summary.IsSolutionUsable();
}

// Output the poses to the file with format: id x y z q_x q_y q_z q_w.
bool OutputPoses(const std::string& filename, const MapOfPoses& poses) {
	std::fstream outfile;
	outfile.open(filename.c_str(), std::istream::out);
	if (!outfile) {
		cout << "Error opening the file: " << filename << endl;
		return false;
	}
	for (std::map<int,
		Pose3d,
		std::less<int>,
		Eigen::aligned_allocator<std::pair<const int, Pose3d> > >::
		const_iterator poses_iter = poses.begin();
		poses_iter != poses.end();
		++poses_iter) {
		const std::map<int,
			Pose3d,
			std::less<int>,
			Eigen::aligned_allocator<std::pair<const int, Pose3d> > >::
			value_type& pair = *poses_iter;
		outfile << pair.first << " " << pair.second.p.transpose() << " "
			<< pair.second.q.x() << " " << pair.second.q.y() << " "
			<< pair.second.q.z() << " " << pair.second.q.w() << '\n';
	}
	return true;
}
