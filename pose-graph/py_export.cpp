#include "py_export.h"
#include "pgo.h"

template<typename T1, typename T2>
static void convert_type(const T1* src, T2* dst, const unsigned int size) {
	for (unsigned int i = 0; i < size; i++)
		dst[i] = (T2)src[i];
}

template<typename T>
static bool check_inf(const T* data, const unsigned int size) {
	for (unsigned int i = 0; i < size; i++)
		if (!isfinite(data[i]))
			return false;
	return true;
}


int py_pose_graph_optm_wrapper(
	const int* poses_idx_pt, const float* poses_pt,
	const int* edges_idx_pt, const float* edges_pose_pt, const float* edges_covar_pt,
	float* poses_ret_pt,
	const int n_poses, const int n_edges,
	const bool optimize_7dof,
	const bool debug) {

	MapOfPoses poses;
	for (int i = 0; i < n_poses; i++) {
		int pose_id = i;
		if (poses_idx_pt)
			pose_id = poses_idx_pt[i];
		if (poses.find(pose_id) != poses.end()) {
			cout << "Duplicate pose id" << endl;
			exit(1);
		}

		double pose_dbl[7];
		convert_type<float, double>(&poses_pt[i * 7], pose_dbl, 7);
		poses[pose_id] = Pose3d(pose_dbl);
	}


	VectorOfConstraints edges;
	for (int i = 0; i < n_edges; i++) {
		int id1 = edges_idx_pt[i * 2 + 0];
		int id2 = edges_idx_pt[i * 2 + 1];
		if (poses.find(id1) == poses.end() || poses.find(id2) == poses.end()) {
			cout << "Invalid edge with non-exist pose id" << endl;
			exit(1);
		}



		double pose_dbl[7];
		double pose_covar_dbl[7 * 7];
		convert_type<float, double>(&edges_pose_pt[i * 7], pose_dbl, 7);
		convert_type<float, double>(&edges_covar_pt[i * 7 * 7], pose_covar_dbl, 7 * 7);

		if (check_inf(pose_covar_dbl, 7 * 7)) {
			edges.push_back(Constraint3d(id1, id2, pose_dbl, pose_covar_dbl));
		}
		else {
			cout << "Warning: Nan/Inf encountered at PGO edge covar" << endl;
		}
	}

	if (debug) {
		cout << "Number of poses: " << poses.size() << endl;
		cout << "Number of constraints: " << edges.size() << endl;
		OutputPoses("poses_original.txt", poses);
	}

	ceres::Problem problem;
	BuildOptimizationProblem(edges, &poses, &problem, optimize_7dof);
	SolveOptimizationProblem(&problem, debug);

	if (debug) {
		OutputPoses("poses_optimized.txt", poses);
	}

	// copy back optimized poses
	for (int i = 0; i < n_poses; i++) {
		int pose_id = i;
		if (poses_idx_pt)
			pose_id = poses_idx_pt[i];
		convert_type<double, float>(poses[pose_id].to_pose7().data(), &poses_ret_pt[i * 7], 7);
	}

	return 0;

}