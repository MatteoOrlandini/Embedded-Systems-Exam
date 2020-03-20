#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>
//#include "alg.h"
//#include "c_math.h"

extern float * host_B;
extern float * host_AUX1;

static const float eps = 1e-4;

int sign_svd (float num){
    if (num > 0) return 1;
	if (num < 0) return -1;
	return 0;
}

int svd_one_sided_jacobi_C(int rows, int columns) {
	// input: B (will be changed), column-major order
	// uses AUX1
	bool exit_flag = false;
	const int M = rows, N = columns;
	int iterations = 0;
	while (!exit_flag) {
		++iterations;
		exit_flag = true;
		for (int j = N - 1; j >= 1; --j)
			for (int i = j - 1; i >= 0; --i) {
				float alpha = 0, beta = 0, gamm = 0;
				float *pi = host_B + M * i, *pj = host_B + M * j;
				for (int k = 0; k < M; ++k) {
					alpha += *pi * *pi;
					beta += *pj * *pj;
					gamm += *pi++ * *pj++;
				}
				if (exit_flag) {
					const float limit = fabsf(gamm) / sqrtf(alpha * beta);
					if (limit > eps) exit_flag = false;
				}
				// some computations (square + square root) need to be done in double precision (64 bits)
				// or accuracy does not reach values comparable to other algorithms
				const float tao = (beta - alpha) / (2 * gamm);
				// t can be computed at 32-bit precision, tests show little loss of accuracy
				//  but good speed improvement
				const float t = sign_svd(tao) / (fabsf(tao) + sqrtf(1 + tao * tao));  // t computed at 32-bit precision
				//const double tao64 = tao;
				//const float t = sign_svd(tao) / (fabsf(tao) + (float)sqrt(1 + tao64 * tao64));  // t computed at 64-bit precision
				// tests show that c must instead be computed at 64-bit precision
				//const float c = 1 / sqrtf(1 + t * t);  // c computed at 32-bit precision
				//const double t64 = t;
				//const float c = 1 / (float)sqrt(1 + t64 * t64);  // c computed at 64-bit precision
				//const float c = 1 / sqrt_1_x2(t);  // results lose an order of magnitude anyway
				const float c = expf(-0.5f * log1pf(t * t));  // new trick by Giorgio! Better than passing to 64 bits.
				const float s = c * t;
				pi = host_B + M * i; pj = host_B + M * j;
				for (int k = 0; k < M; ++k) {
					const float t = *pi;
					*pi++ = c * t - s * *pj;
					*pj = s * t + c * *pj;
					++pj;
				}
			}
	}
	for (int j = 0; j < N; ++j) {
		float t = 0, *pj = host_B + M * j;
		for (int k = 0; k < M; ++k, ++pj) t += *pj * *pj;
		host_AUX1[j] = sqrtf(t);
	}
	return iterations;
}

