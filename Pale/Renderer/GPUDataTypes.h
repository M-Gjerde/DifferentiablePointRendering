//
// Created by magnus on 8/29/25.
//
#pragma once

#include <glm/glm.hpp>
#include <sycl/sycl.hpp>

namespace Pale {
    // --- Generic M×N matrix wrapping sycl::vec<float,N> rows ---

    // --- Generic M×N matrix wrapping sycl::vec<float,N> rows ---
    template<size_t M, size_t N>
    struct Matrix {
        static_assert(M > 0 && N > 0, "Matrix dimensions must be positive");
        using RowType = sycl::vec<float, N>;
        using value_type = float;
        std::array<RowType, M> row;

        // default constructor
        Matrix() = default;

        // cast constructor: drop extra cols/rows if converting from larger matrix
        template<size_t P, size_t Q,
            typename = std::enable_if_t<(P >= M && Q >= N)> >
        explicit Matrix(Matrix<P, Q> const &other) {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    row[i][j] = other.row[i][j];
        }
    };

    // --- Matrix × Matrix multiplication ---
    template<size_t M, size_t N, size_t P>
    Matrix<M, P> operator*(Matrix<M, N> const &A,
                           Matrix<N, P> const &B) {
        Matrix<M, P> C{};
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < P; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < N; ++k)
                    sum += A.row[i][k] * B.row[k][j];
                C.row[i][j] = sum;
            }
        }
        return C;
    }

    // --- Matrix × Vector (MxN × N) → M-vector ---
    template<size_t M, size_t N>
    sycl::vec<float, M> operator*(Matrix<M, N> const &A,
                                  sycl::vec<float, N> const &v) {
        static_assert(M > 0 && M <= 16, "M must be in [1,16]");
        static_assert(N > 0 && N <= 16, "N must be in [1,16]");
        sycl::vec<float, M> r{};
        for (size_t i = 0; i < M; ++i) {
            // Use dot product to avoid any out-of-bounds indexing
            r[i] = sycl::dot(A.row[i], v);
        }
        return r;
    }

    // --- Vector × Matrix (M × MxN) → N-vector ---
    template<size_t M, size_t N>
    sycl::vec<float, N> operator*(sycl::vec<float, M> const &v,
                                  Matrix<M, N> const &A) {
        sycl::vec<float, N> r{};
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < M; ++i)
                sum += v[i] * A.row[i][j];
            r[j] = sum;
        }
        return r;
    }

    // --- Scalar × Matrix multiplication ---
    template<size_t M, size_t N>
    Matrix<M, N> operator*(Matrix<M, N> const &A, float s) {
        Matrix<M, N> R{};
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                R.row[i][j] = A.row[i][j] * s;
        return R;
    }

    template<size_t M, size_t N>
    Matrix<M, N> operator*(float s, Matrix<M, N> const &A) {
        return A * s;
    }


    // --- Transpose ---
    template<size_t M, size_t N>
    Matrix<N, M> transpose(Matrix<M, N> const &m) {
        Matrix<N, M> t{};
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                t.row[j][i] = m.row[i][j];
        return t;
    }

    // --- Inverse for 3×3 ---
    inline Matrix<3, 3> inverse(Matrix<3, 3> const &m) {
        auto &r = m.row;
        float a00 = r[0][0], a01 = r[0][1], a02 = r[0][2];
        float a10 = r[1][0], a11 = r[1][1], a12 = r[1][2];
        float a20 = r[2][0], a21 = r[2][1], a22 = r[2][2];
        float co0 = a11 * a22 - a12 * a21;
        float co1 = -a10 * a22 + a12 * a20;
        float co2 = a10 * a21 - a11 * a20;
        float det = a00 * co0 + a01 * co1 + a02 * co2;
        Matrix<3, 3> inv{};
        inv.row[0] = sycl::vec<float, 3>(co0, (-a01 * a22 + a02 * a21), (a01 * a12 - a02 * a11));
        inv.row[1] = sycl::vec<float, 3>(co1, (a00 * a22 - a02 * a20), (-a00 * a12 + a02 * a10));
        inv.row[2] = sycl::vec<float, 3>(co2, (-a00 * a21 + a01 * a20), (a00 * a11 - a01 * a10));
        return inv * (1.0f / det);
    }


    /* ---------- 1. POD wrapper, no inheritance, no namespace injection -------- */
    struct alignas(16) float3 {
        sycl::vec<float, 3> v __attribute__((aligned(16)));

        /* implicit from base */
        float3(sycl::vec<float, 3> const &b = {0, 0, 0}) : v(b) {
        }

        float3(float x, float y, float z) : v{x, y, z} {
        }

        explicit float3(float scalar) : v{scalar, scalar, scalar} {
        }

        explicit float3(sycl::vec<float, 4> const &b) : v{b.x(), b.y(), b.z()} {
        }

        /* implicit to base */
        operator sycl::vec<float, 3>() const { return v; }

        /* ---------- subscript operator ----------------------------------- */
        float &operator[](std::size_t i) { return v[i]; } // l-value
        float operator[](std::size_t i) const { return v[i]; } // r-value

        /* ---------- unary operators ------------------------------------ */
        float3 operator-() const { return float3{-v}; }
        float3 operator+() const { return *this; } // optional

        float3 &operator=(const glm::vec3 &c) noexcept {
            v = {c.x, c.y, c.z};
            return *this;
        }

        float3 &operator=(const float &val) noexcept {
            v = {val, val, val};
            return *this;
        }

        /* helpers identical to sycl::vec API */
        float x() const { return v.x(); }
        float &x() { return v.x(); }
        float y() const { return v.y(); }
        float &y() { return v.y(); }
        float z() const { return v.z(); }
        float &z() { return v.z(); }

        bool operator==(const float3 &float3) const {
            return v.x() == float3.x() && v.y() == float3.y() && v.z() == float3.z();
        };
    };

    /* float4, same idea -------------------------------------------------------- */
    struct float4 {
        sycl::vec<float, 4> v;
        /* implicit */
        /* NOLINTNEXTLINE(google-explicit-constructor) */
        float4(sycl::vec<float, 4> const &b = {0, 0, 0, 0}) : v(b) {
        }

        float4(float x, float y, float z, float w) : v{x, y, z, w} {
        }

        explicit float4(float scalar) : v{scalar, scalar, scalar, scalar} {
        }

        float4(float3 const &p, float w) : v{p.x(), p.y(), p.z(), w} {
        }

        operator sycl::vec<float, 4>() const { return v; }

        /* ---------- subscript operator ----------------------------------- */
        float &operator[](std::size_t i) { return v[i]; } // l-value
        float operator[](std::size_t i) const { return v[i]; } // r-value

        float x() const { return v.x(); }
        float &x() { return v.x(); }
        float y() const { return v.y(); }
        float &y() { return v.y(); }
        float z() const { return v.z(); }
        float &z() { return v.z(); }
        float w() const { return v.w(); }
        float &w() { return v.w(); }
    };

    /* ---------- 2. arithmetic that delegates to sycl::vec --------------------- */
    inline float3 operator+(float3 a, float3 b) {
        return {
            a.x() + b.x(),
            a.y() + b.y(),
            a.z() + b.z()
        };
    }

    inline float3 operator-(float3 a, float3 b) {
        return {
            a.x() - b.x(),
            a.y() - b.y(),
            a.z() - b.z()
        };
    }

    /* component-wise product */
    inline float3 operator*(float3 a, float3 b) {
        return {
            a.x() * b.x(),
            a.y() * b.y(),
            a.z() * b.z()
        };
    }

    /* scalar products */
    inline float3 operator*(float3 a, float s) {
        return {a.x() * s, a.y() * s, a.z() * s};
    }

    inline float3 operator*(float s, float3 a) { return a * s; }

    inline float3 operator/(float3 a, float s) {
        float inv = 1.f / s;
        return {a.x() * inv, a.y() * inv, a.z() * inv};
    }


    inline float3 min(float3 a, float3 b) { return sycl::min(a.v, b.v); }
    inline float min(float a, float b) { return sycl::min(a, b); }
    inline float3 max(float3 a, float3 b) { return sycl::max(a.v, b.v); }
    inline float max(float a, float b) { return sycl::max(a, b); }
    inline float3 cross(float3 a, float3 b) { return sycl::cross(a.v, b.v); }
    inline float dot(float3 a, float3 b) { return sycl::dot(a.v, b.v); }
    inline float length(float3 a) { return sycl::length(a.v); }


    inline float4 operator/(float4 a, float s) {
        float inv = 1.f / s;
        return {a.x() * inv, a.y() * inv, a.z() * inv, a.w() * inv};
    }

    inline float4 operator+(float4 a, float4 b) {
        return {
            a.x() + b.x(),
            a.y() + b.y(),
            a.z() + b.z(),
            a.w() + b.w()
        };
    }



    /*
    // --- Normalize vector ---
    inline float4 normalize(float4 &v) {
        float sum = 0.0f;
        for (size_t i = 0; i < 4; ++i)
            sum += v[i] * v[i];
        float invLen = sycl::rsqrt(sum);
        return v * invLen;
    }
    */
    inline float3 normalize(const float3 &v) {
        float sum = 0.0f;
        for (size_t i = 0; i < 3; ++i)
            sum += v[i] * v[i];
        float invLen = sycl::rsqrt(sum);
        return v * invLen;
    }

    /* keep float2 as single alias once */
    using float2 = sycl::float2;

    // --- Convenient aliases ---
    using float2x2 = Matrix<2, 2>;
    using float3x3 = Matrix<3, 3>;
    using float4x4 = Matrix<4, 4>;


    // --- Wrapper overloads to route through base operator* ---
    template<size_t M>
    sycl::vec<float, M> operator*(Matrix<M, 3> const &A, float3 const &v) {
        return operator*<M, 3>(A, static_cast<sycl::vec<float, 3>>(v));
    }

    template<size_t M>
    sycl::vec<float, M> operator*(Matrix<M, 4> const &A, float4 const &v) {
        // explicitly invoke the Matrix×Vector template with N=4
        return operator*<M, 4>(A, static_cast<sycl::vec<float, 4>>(v));
    }

    static_assert(sizeof(float3) == 16, "float3 must be 16 bytes");
    static_assert(alignof(float3) == 16, "float3 must be 16-byte aligned");
    static_assert(sizeof(float3x3) == 3 * sizeof(sycl::vec<float, 3>),
                  "float3x3 layout must stay row-major");
    static_assert(sizeof(float4x4) == 4 * sizeof(sycl::vec<float, 4>),
                  "float4x4 layout must stay row-major");

    /*------------- Make a 3×3 from the top-left of a 4×4 ------------------*/
    inline float3x3 linearPart(float4x4 const &m4) {
        float3x3 m3;
        for (int i = 0; i < 3; ++i) // rows
            for (int j = 0; j < 3; ++j) // cols
                m3.row[i][j] = m4.row[i][j];
        return m3;
    }

    /*------------- Determinant of a 3×3 (needed for handedness test) ------*/
    inline float det(float3x3 const &m) {
        auto &r = m.row;
        return
                r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) -
                r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) +
                r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    }


    /*------------- Transform a normal ------------------------------------*/
    inline float3 transformNormal(float3 const &n_obj, float4x4 const &obj2world) {
        float3x3 M = linearPart(obj2world); // upper-left 3×3
        float3x3 N = transpose(inverse(M)); // (M⁻¹)ᵀ
        float3 n = normalize(N * n_obj); // renormalise!

        /* Optional: flip if the instance changes handedness */
        if (det(M) < 0.0f) n = -n;

        return n;
    }


    template<int MaxN = 256>
    struct SmallStack {
        uint32_t data[MaxN];
        int sp = 0;

        bool push(uint32_t v) // returns false on overflow
        {
            if (sp >= MaxN) return false;
            data[sp++] = v;
            return true;
        }

        uint32_t pop() // safe pop – returns 0 on underflow
        {
            if (sp <= 0) return 0u;
            return data[--sp];
        }

        bool empty() const { return sp == 0; }
    };

    struct LeafRange {
        uint32_t node, first, count;
    };

    inline float4x4 glm2sycl(const glm::mat4 &m) {
        float4x4 out;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                out.row[r][c] = m[c][r];
        return out;
    }

    inline float3x3 glm2sycl(const glm::mat3 &m) {
        float3x3 out;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                out.row[r][c] = m[c][r];
        return out;
    }

    // --- Outer product for float3 (a * b^T) → 3×3 matrix ---
    inline float3x3 outerProduct(const float3 &leftVector, const float3 &rightVector) {
        float3x3 resultMatrix{};
        const float ax = leftVector.x();
        const float ay = leftVector.y();
        const float az = leftVector.z();

        const float bx = rightVector.x();
        const float by = rightVector.y();
        const float bz = rightVector.z();

        // Row 0 = a_x * [b_x, b_y, b_z]
        resultMatrix.row[0] = sycl::vec<float, 3>(ax * bx, ax * by, ax * bz);
        // Row 1 = a_y * [b_x, b_y, b_z]
        resultMatrix.row[1] = sycl::vec<float, 3>(ay * bx, ay * by, ay * bz);
        // Row 2 = a_z * [b_x, b_y, b_z]
        resultMatrix.row[2] = sycl::vec<float, 3>(az * bx, az * by, az * bz);

        return resultMatrix;
    }

    // --- Matrix + and - ---
template<size_t M, size_t N>
inline Matrix<M, N> operator+(const Matrix<M, N> &leftMatrix,
                              const Matrix<M, N> &rightMatrix) {
    Matrix<M, N> resultMatrix{};
    for (size_t rowIndex = 0; rowIndex < M; ++rowIndex)
        for (size_t colIndex = 0; colIndex < N; ++colIndex)
            resultMatrix.row[rowIndex][colIndex] =
                leftMatrix.row[rowIndex][colIndex] + rightMatrix.row[rowIndex][colIndex];
    return resultMatrix;
}

template<size_t M, size_t N>
inline Matrix<M, N> operator-(const Matrix<M, N> &leftMatrix,
                              const Matrix<M, N> &rightMatrix) {
    Matrix<M, N> resultMatrix{};
    for (size_t rowIndex = 0; rowIndex < M; ++rowIndex)
        for (size_t colIndex = 0; colIndex < N; ++colIndex)
            resultMatrix.row[rowIndex][colIndex] =
                leftMatrix.row[rowIndex][colIndex] - rightMatrix.row[rowIndex][colIndex];
    return resultMatrix;
}

// --- Scalar / Matrix (convenience) ---
template<size_t M, size_t N>
inline Matrix<M, N> operator/(const Matrix<M, N> &inputMatrix, float scalarValue) {
    float inverseScalar = 1.0f / scalarValue;
    return inputMatrix * inverseScalar;
}

// --- Identity matrices ---
template<size_t N>
inline Matrix<N, N> identityMatrix() {
    Matrix<N, N> identity{};
    for (size_t rowIndex = 0; rowIndex < N; ++rowIndex) {
        for (size_t colIndex = 0; colIndex < N; ++colIndex)
            identity.row[rowIndex][colIndex] = (rowIndex == colIndex) ? 1.0f : 0.0f;
    }
    return identity;
}

inline float3x3 identity3x3() { return identityMatrix<3>(); }

    // --- Matrix × float3 → float3 (3×3 case) ---
    inline float3 operator*(const float3x3 &leftMatrix, const float3 &rightVector) {
        sycl::vec<float, 3> productResult =
            operator*<3, 3>(leftMatrix, static_cast<sycl::vec<float, 3>>(rightVector));
        return float3{productResult};
    }

    // --- Matrix × float4 → float4 (4×4 case) ---
    inline float4 operator*(const float4x4 &leftMatrix, const float4 &rightVector) {
        sycl::vec<float, 4> productResult =
            operator*<4, 4>(leftMatrix, static_cast<sycl::vec<float, 4>>(rightVector));
        return float4{productResult};
    }

    // --- float3 × Matrix → float3 (row-vector form, 3×3) ---
    inline float3 operator*(const float3 &leftVector, const float3x3 &rightMatrix) {
        sycl::vec<float, 3> productResult =
            operator*<3, 3>(static_cast<sycl::vec<float, 3>>(leftVector), rightMatrix);
        return float3{productResult};
    }

    // --- float4 × Matrix → float4 (row-vector form, 4×4) ---
    inline float4 operator*(const float4 &leftVector, const float4x4 &rightMatrix) {
        sycl::vec<float, 4> productResult =
            operator*<4, 4>(static_cast<sycl::vec<float, 4>>(leftVector), rightMatrix);
        return float4{productResult};
    }

}
