#pragma once

#include "solve_cubic.h"
#include "solve_eig0.h"
#include "refine_lambda.h"
#include "matrix.h"
#include <algorithm>
#include <iostream>





namespace cvl {




	template<class T, int refinement_iterations = 5>
	__mlib_host_device
		int p3p_lambdatwist(Vector3<T> y1,
			Vector3<T> y2,
			Vector3<T> y3,
			Vector3<T> x1,
			Vector3<T> x2,
			Vector3<T> x3,
			Vector<cvl::Matrix<T, 3, 3>, 4>& Rs,
			Vector<Vector3<T>, 4>& Ts) {





		// normalize the length of ys, we could expect it, but lets not...
		y1.normalize();
		y2.normalize();
		y3.normalize();


		T b12 = -2.0*(y1.dot(y2));
		T b13 = -2.0*(y1.dot(y3));
		T b23 = -2.0*(y2.dot(y3));




		// implicit creation of Vector3<T> can be removed
		Vector3<T> d12 = x1 - x2;
		Vector3<T> d13 = x1 - x3;
		Vector3<T> d23 = x2 - x3;
		Vector3<T> d12xd13(d12.cross(d13));



		T a12 = d12.squaredLength();
		T a13 = d13.squaredLength();
		T a23 = d23.squaredLength();

		//if(abs(D1.determinant())<1e-5 || fabs(D2.determinant())<1e-5)        cout<<"det(D): "<<D1.determinant()<<" "<<D2.determinant()<<endl;



		//a*g^3 + b*g^2 + c*g + d = 0
		T c31 = -0.5*b13;
		T c23 = -0.5*b23;
		T c12 = -0.5*b12;
		T blob = (c12*c23*c31 - 1.0);

		T s31_squared = 1.0 - c31 * c31;
		T s23_squared = 1.0 - c23 * c23;
		T s12_squared = 1.0 - c12 * c12;



		T p3 = (a13*(a23*s31_squared - a13 * s23_squared));

		T p2 = 2.0*blob*a23*a13 + a13 * (2.0*a12 + a13)*s23_squared + a23 * (a23 - a12)*s31_squared;

		T p1 = a23 * (a13 - a23)*s12_squared - a12 * a12*s23_squared - 2.0*a12*(blob*a23 + a13 * s23_squared);

		T p0 = a12 * (a12*s23_squared - a23 * s12_squared);


		T g = 0;

		//p3 is det(D2) so its definietly >0 or its a degenerate case

		{
			p3 = 1.0 / p3;
			p2 *= p3;
			p1 *= p3;
			p0 *= p3;

			// get sharpest real root of above...

			g = cubick(p2, p1, p0);
		}





		// we can swap D1,D2 and the coeffs!
		// oki, Ds are:
		//D1=M12*XtX(2,2) - M23*XtX(1,1);
		//D2=M23*XtX(3,3) - M13*XtX(2,2);

		//[    a23 - a23*g,                 (a23*b12)/2,              -(a23*b13*g)/2]
		//[    (a23*b12)/2,           a23 - a12 + a13*g, (a13*b23*g)/2 - (a12*b23)/2]
		//[ -(a23*b13*g)/2, (a13*b23*g)/2 - (a12*b23)/2,         g*(a13 - a23) - a12]


		// gain 13 ns...
		T A00 = a23 * (1.0 - g);
		T A01 = (a23*b12)*0.5;
		T A02 = (a23*b13*g)*(-0.5);
		T A11 = a23 - a12 + a13 * g;
		T A12 = b23 * (a13*g - a12)*0.5;
		T A22 = g * (a13 - a23) - a12;



		Matrix<T, 3, 3> A(A00, A01, A02,
			A01, A11, A12,
			A02, A12, A22);




		// get sorted eigenvalues and eigenvectors given that one should be zero...
		Matrix<T, 3, 3> V;
		Vector3<T> L;
		eigwithknown0(A, V, L);


		//T v = std::sqrt(std::max(T(0), -L(1) / L(0)));
		T v = std::sqrt(-L(1) / L(0) > 0 ? -L(1) / L(0) : T(0));




		int valid = 0;
		Vector<Vector<T, 3>, 4> Ls;


		// use the t=Vl with t2,st2,t3 and solve for t3 in t2
		{ //+v

			T s = v;
			//T w2=T(1.0)/( s*V(0,1) - V(0,0));
			//T w0=(V(1,0) - s*V(1,1))*w2;
			//T w1=(V(2,0) - s*V(2,1))*w2;

			T w2 = T(1.0) / (s*V(1) - V(0));
			T w0 = (V(3) - s * V(4))*w2;
			T w1 = (V(6) - s * V(7))*w2;



			T a = T(1.0) / ((a13 - a12)*w1*w1 - a12 * b13*w1 - a12);
			T b = (a13*b12*w1 - a12 * b13*w0 - T(2.0)*w0*w1*(a12 - a13))*a;
			T c = ((a13 - a12)*w0*w0 + a13 * b12*w0 + a13)*a;



			if (b*b - 4.0*c >= 0) {
				T tau1, tau2;
				root2real(b, c, tau1, tau2);
				if (tau1 > 0) {
					T tau = tau1;
					T d = a23 / (tau*(b23 + tau) + T(1.0));

					T l2 = std::sqrt(d);
					T l3 = tau * l2;

					T l1 = w0 * l2 + w1 * l3;
					if (l1 >= 0) {

						Ls[valid] = { l1,l2,l3 };

						++valid;
					}

				}
				if (tau2 > 0) {
					T tau = tau2;
					T d = a23 / (tau*(b23 + tau) + T(1.0));

					T l2 = std::sqrt(d);
					T l3 = tau * l2;
					T l1 = w0 * l2 + w1 * l3;
					if (l1 >= 0) {
						Ls[valid] = { l1,l2,l3 };
						++valid;
					}

				}
			}
		}

		{ //+v
			T s = -v;
			T w2 = T(1.0) / (s*V(0, 1) - V(0, 0));
			T w0 = (V(1, 0) - s * V(1, 1))*w2;
			T w1 = (V(2, 0) - s * V(2, 1))*w2;

			T a = T(1.0) / ((a13 - a12)*w1*w1 - a12 * b13*w1 - a12);
			T b = (a13*b12*w1 - a12 * b13*w0 - T(2.0)*w0*w1*(a12 - a13))*a;
			T c = ((a13 - a12)*w0*w0 + a13 * b12*w0 + a13)*a;


			if (b*b - 4.0*c >= 0) {
				T tau1, tau2;

				root2real(b, c, tau1, tau2);
				if (tau1 > 0) {
					T tau = tau1;
					T d = a23 / (tau*(b23 + tau) + T(1.0));
					if (d > 0) {
						T l2 = std::sqrt(d);

						T l3 = tau * l2;

						T l1 = w0 * l2 + w1 * l3;
						if (l1 >= 0) {
							Ls[valid] = { l1,l2,l3 };
							++valid;
						}
					}
				}
				if (tau2 > 0) {
					T tau = tau2;
					T d = a23 / (tau*(b23 + tau) + T(1.0));
					if (d > 0) {
						T l2 = std::sqrt(d);

						T l3 = tau * l2;

						T l1 = w0 * l2 + w1 * l3;
						if (l1 >= 0) {
							Ls[valid] = { l1,l2,l3 };
							++valid;
						}
					}
				}
			}
		}



		for (int i = 0; i < valid; ++i) { gauss_newton_refineL<T, refinement_iterations>(Ls[i], a12, a13, a23, b12, b13, b23); }

		Vector3<T> ry1, ry2, ry3;
		Vector3<T> yd1;
		Vector3<T> yd2;
		Vector3<T> yd1xd2;
		Matrix<T, 3, 3> X(d12(0), d13(0), d12xd13(0),
			d12(1), d13(1), d12xd13(1),
			d12(2), d13(2), d12xd13(2));
		X = X.inverse();

		for (int i = 0; i < valid; ++i) {
			//cout<<"Li="<<Ls(i)<<endl;

			// compute the rotation:
			ry1 = y1 * Ls(i)(0);
			ry2 = y2 * Ls(i)(1);
			ry3 = y3 * Ls(i)(2);

			yd1 = ry1 - ry2;
			yd2 = ry1 - ry3;
			yd1xd2 = yd1.cross(yd2);

			Matrix<T, 3, 3> Y(yd1(0), yd2(0), yd1xd2(0),
				yd1(1), yd2(1), yd1xd2(1),
				yd1(2), yd2(2), yd1xd2(2));


			Rs[i] = Y * X;

			Ts[i] = (ry1 - Rs[i] * x1);

		}


		return valid;







	}
}
