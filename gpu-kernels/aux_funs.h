#pragma once

double inverse(double* mat, double* mat_inv, int N);

double determinant(double* mat, int N);

double regularize_covar_LW_given_lambda(double* mat, double* mat_ret, double lambda, int dims);
