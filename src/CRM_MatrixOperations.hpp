#pragma once

namespace CRMCatheterModel {

	// all matrices are assumed to be stored as a one dimensional array in row major order, unless otherwise noted

	//matrix multiplication C=A*B  ( A:D1xD2 B:D2xD3 gives C:D1xD3 )
	template <int D1, int D2, int D3, typename AScalar, typename BScalar, typename CScalar>
	inline void mMult_AB(const AScalar in_A[D1 * D2], const BScalar in_B[D2 * D3], CScalar out_C[D1 * D3]) {

		for (int i = 0; i < D1; i++) {
			for (int j = 0; j < D3; j++) {
				CScalar sum = CScalar(0);
				for (int k = 0; k < D2; k++) {
					sum += CScalar(in_A[i * D2 + k]) * CScalar(in_B[k * D3 + j]);
				}
				out_C[i * D3 + j] = sum;
			}
		}

	}


	//matrix multiplication C=A^T*B  (A transposed times B)  ( A:D1xD2, A^T:D2xD1, B:D1xD3 gives C:D2xD3 )
	template <int D1, int D2, int D3, typename AScalar, typename BScalar, typename CScalar>
	inline void mMult_ATB(const AScalar in_A[D1 * D2], const BScalar in_B[D1 * D3], CScalar out_C[D2 * D3]) {

		for (int i = 0; i < D2; i++) {
			for (int j = 0; j < D3; j++) {
				CScalar sum = CScalar(0);
				for (int k = 0; k < D1; k++) {
					sum += CScalar(in_A[k * D2 + i]) * CScalar(in_B[k * D3 + j]);
				}
				out_C[i * D3 + j] = sum;
			}
		}

	}


	//matrix multiplication C=A*B^T  (A times B transposed)  ( A:D1xD2, B:D3xD2, B^T:D2xD3,  gives C:D1xD3 )
	template <int D1, int D2, int D3, typename AScalar, typename BScalar, typename CScalar>
	inline void mMult_ABT(const AScalar in_A[D1 * D2], const BScalar in_B[D3 * D2], CScalar out_C[D1 * D3]) {

		for (int i = 0; i < D1; i++) {
			for (int j = 0; j < D3; j++) {
				CScalar sum = CScalar(0);
				for (int k = 0; k < D2; k++) {
					sum += CScalar(in_A[i * D2 + k]) * CScalar(in_B[j * D2 + k]);
				}
				out_C[i * D3 + j] = sum;
			}
		}

	}


	// matrix scalar multiplication C=s*A  ( s:scalar, A,C:D1xD2 )
	template <int D1, int D2, typename SScalar, typename AScalar, typename CScalar>
	inline void mMult_sA(const SScalar s, const AScalar in_A[D1 * D2], CScalar out_C[D1 * D2]) {

		for (int i = 0; i < D1; i++) {
			for (int j = 0; j < D2; j++) {
				out_C[i * D2 + j] = CScalar(s) * CScalar(in_A[i * D2 + j]);
			}
		}

	}


	//matrix multiply and add C=C+A*B  ( A:D1xD2 B:D2xD3 gives C:D1xD3 )
	template <int D1, int D2, int D3, typename AScalar, typename BScalar, typename CScalar>
	inline void mMultAdd_AB(const AScalar in_A[D1 * D2], const BScalar in_B[D2 * D3], CScalar out_C[D1 * D3]) {

		for (int i = 0; i < D1; i++) {
			for (int j = 0; j < D3; j++) {
				CScalar sum = CScalar(0);
				for (int k = 0; k < D2; k++) {
					sum += CScalar(in_A[i * D2 + k]) * CScalar(in_B[k * D3 + j]);
				}
				out_C[i * D3 + j] += sum;
			}
		}

	}


	//matrix multiply and subtract C=C-A*B  ( A:D1xD2 B:D2xD3 gives C:D1xD3 )
	template <int D1, int D2, int D3, typename AScalar, typename BScalar, typename CScalar>
	inline void mMultSub_AB(const AScalar in_A[D1 * D2], const BScalar in_B[D2 * D3], CScalar out_C[D1 * D3]) {

		for (int i = 0; i < D1; i++) {
			for (int j = 0; j < D3; j++) {
				CScalar sum = CScalar(0);
				for (int k = 0; k < D2; k++) {
					sum += CScalar(in_A[i * D2 + k]) * CScalar(in_B[k * D3 + j]);
				}
				out_C[i * D3 + j] -= sum;
			}
		}

	}


	// matrix addition X=A+B, A=A+B, X=A+sB,  or X=A+B+C   ( A,B,C,X:D1xD2, s:scalar )
	template <int D1, int D2, typename AScalar, typename BScalar, typename XScalar>
	inline void mAdd_AB(const AScalar in_A[D1 * D2], const BScalar in_B[D1 * D2], XScalar out_X[D1 * D2]) {

		for (int i = 0; i < D1; i++) {
			for (int j = 0; j < D2; j++) {
				out_X[i * D2 + j] = XScalar(in_A[i * D2 + j]) + XScalar(in_B[i * D2 + j]);
			}
		}

	}

	template <int D1, int D2, typename AScalar, typename BScalar>
	inline void mAdd_AB(AScalar inout_A[D1 * D2], const BScalar in_B[D1 * D2]) {

		for (int i = 0; i < D1; i++) {
			for (int j = 0; j < D2; j++) {
				inout_A[i * D2 + j] = AScalar(inout_A[i * D2 + j]) + AScalar(in_B[i * D2 + j]);
			}
		}

	}

	template <int D1, int D2, typename AScalar, typename SScalar, typename BScalar, typename XScalar>
	inline void mAdd_AsB(const AScalar in_A[D1 * D2], const SScalar in_s, const BScalar in_B[D1 * D2], XScalar out_X[D1 * D2]) {

		for (int i = 0; i < D1; i++) {
			for (int j = 0; j < D2; j++) {
				out_X[i * D2 + j] = XScalar(in_A[i * D2 + j]) + XScalar(in_s) * XScalar(in_B[i * D2 + j]);
			}
		}

	}

	template <int D1, int D2, typename AScalar, typename BScalar, typename CScalar, typename XScalar>
	inline void mAdd_ABC(const AScalar in_A[D1 * D2], const BScalar in_B[D1 * D2], const CScalar in_C[D1 * D2], XScalar out_X[D1 * D2]) {

		for (int i = 0; i < D1; i++) {
			for (int j = 0; j < D2; j++) {
				out_X[i * D2 + j] = XScalar(in_A[i * D2 + j]) + XScalar(in_B[i * D2 + j]) + XScalar(in_C[i * D2 + j]);
			}
		}

	}


	// matrix subtraction X=A-B   ( A,B,X:D1xD2 )
	template <int D1, int D2, typename AScalar, typename BScalar, typename XScalar>
	inline void mSub_AB(const AScalar in_A[D1 * D2], const BScalar in_B[D1 * D2], XScalar out_X[D1 * D2]) {

		for (int i = 0; i < D1; i++) {
			for (int j = 0; j < D2; j++) {
				out_X[i * D2 + j] = XScalar(in_A[i * D2 + j]) - XScalar(in_B[i * D2 + j]);
			}
		}

	}


	// matrix subtraction A=A-B   ( A,B:D1xD2 )
	template <int D1, int D2, typename AScalar, typename BScalar>
	inline void mSub_AB(AScalar inout_A[D1 * D2], const BScalar in_B[D1 * D2]) {

		for (int i = 0; i < D1; i++) {
			for (int j = 0; j < D2; j++) {
				inout_A[i * D2 + j] = AScalar(inout_A[i * D2 + j]) - AScalar(in_B[i * D2 + j]);
			}
		}

	}


	// 2-norm of a vector  ( D1x1 vector )
	template <int D1, typename VScalar, typename OutScalar = VScalar>
	inline OutScalar vNormSq(const VScalar v[D1]) {

		OutScalar vn = OutScalar(0);

		for (int i = 0; i < D1; i++) {
			vn += OutScalar(v[i]) * OutScalar(v[i]);
		}
		return vn;

	}


	//copy matrix: B=A  (D1xD2 matrices)
	template <int D1, int D2, typename AScalar, typename BScalar>
	inline void mCopy_ABm(const AScalar in_A[D1][D2], BScalar out_B[D1][D2]) {

		for (int i = 0; i < D1; i++) {
			for (int j = 0; j < D2; j++) {
				out_B[i][j] = BScalar(in_A[i][j]);
			}
		}

	}



	//copy vector: B=A  (D1x1 vector)
	template <int D1, typename AScalar, typename BScalar>
	inline void mCopy_AB(const AScalar in_A[D1], BScalar out_B[D1]) {

		for (int i = 0; i < D1; i++) {
			out_B[i] = BScalar(in_A[i]);
		}

	}

}
