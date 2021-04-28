//#define EIGEN_IMPL  // eigen has compiling issues with nvcc...
#ifdef EIGEN_IMPL
#include "aux_funs.h"
#include <Eigen/Dense>

using namespace Eigen;

double inverse(double* mat, double* mat_inv, int N) {
	Map<MatrixXd> a(mat, N, N);

	double deter = a.determinant();
	if (deter > 0) {
		a = a.inverse();

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				mat_inv[i*N + j] = a(i, j);
	}

	return deter;
}

double determinant(double* mat, int N) {
	Map<MatrixXd> a(mat, N, N);
	return a.determinant();
}
/*
// Ledoit-Wolf shrinkage estimator http://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf
// Some other documents https://statistics.stanford.edu/sites/g/files/sbiybj6031/f/2012-10.pdf
double regularize_covar_LW(double* mat, double* mat_ret, double b2_bar, int dims) {
	MatrixXd S(dims, dims);
	for (int i = 0; i < dims; i++)
		for (int j = 0; j < dims; j++)
			S(i, j) = mat[i*dims + j];
	//printf("det before = %e \N", S.determinant());
	MatrixXd I(dims, dims);
	I.setIdentity();
	double m = S.trace() / (double)dims;
	double d2 = (S - m * I).squaredNorm();
	double b2 = std::min(b2_bar, d2);

	double lambda = b2 / d2;
	//printf("lambda = %lf\n", lambda);
	MatrixXd S_star;
	S_star = lambda * m*I + (1 - lambda)*S;


	for (int i = 0; i < dims; i++)
		for (int j = 0; j < dims; j++)
			mat_ret[i*dims + j] = S_star(i, j);
	//printf("det after = %e \N", S.determinant());
	return S.determinant();
}
*/

// Ledoit-Wolf shrinkage estimator http://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf
// Some other documents https://statistics.stanford.edu/sites/g/files/sbiybj6031/f/2012-10.pdf
double regularize_covar_LW_given_lambda(double* mat, double* mat_ret, double lambda, int dims) {
	Map<MatrixXd> S(mat, dims, dims);
	
	//printf("det before = %e \n", S.determinant());
	MatrixXd I(dims, dims);
	I.setIdentity();
	double m = S.trace() / (double)dims;

	//printf("lambda = %lf\n", lambda);
	MatrixXd S_star;
	S_star = lambda * m*I + (1 - lambda)*S;

	for (int i = 0; i < dims; i++)
		for (int j = 0; j < dims; j++)
			mat_ret[i*dims + j] = S_star(i, j);
	//printf("det after = %e \n", S_star.determinant());

#if 0
	JacobiSVD<MatrixXd> S_svd(S, ComputeFullU | ComputeFullV);
	VectorXd S_s = S_svd.singularValues();
	JacobiSVD<MatrixXd> S_star_svd(S_star, ComputeFullU | ComputeFullV);
	VectorXd S_star_s = S_star_svd.singularValues();


	printf("before : ");
	for (int d = 0; d < dims; d++)
		printf("%lf ", S_s(d));
	printf("\n");

	printf("after : ");
	for (int d = 0; d < dims; d++)
		printf("%lf ", S_star_s(d));
	printf("\n");
#endif
	return S_star.determinant();
}

#else

#include "aux_funs.h"
#include <opencv2/highgui.hpp>


double inverse(double* mat, double* mat_inv, int N) {
	cv::Matx66d a(mat);

	double deter = cv::determinant(a);
	if (deter > 0) {
		a = a.inv();

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				mat_inv[i*N + j] = a(i, j);
	}

	return deter;
}

double determinant(double* mat, int N) {
	cv::Matx66d a(mat);
	return cv::determinant(a);
}

// Ledoit-Wolf shrinkage estimator http://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf
// Some other documents https://statistics.stanford.edu/sites/g/files/sbiybj6031/f/2012-10.pdf
double regularize_covar_LW_given_lambda(double* mat, double* mat_ret, double lambda, int dims) {
	cv::Matx66d S(mat);

	//printf("det before = %e \n", S.determinant());
	cv::Matx66d I= cv::Matx66d::eye();

	double m = cv::trace(S) / (double)dims;

	//printf("lambda = %lf\n", lambda);
	cv::Matx66d S_star;
	S_star = lambda * m*I + (1 - lambda)*S;

	for (int i = 0; i < dims; i++)
		for (int j = 0; j < dims; j++)
			mat_ret[i*dims + j] = S_star(i, j);
	//printf("det after = %e \n", S_star.determinant());

	return cv::determinant(S_star);
}
#endif