#pragma once

#include "Renderer/GPUDataStructures.h"

namespace Pale {

// A camera-independent chart, reconstructed from two existing elliptical surfels.
// The polynomial intersection uses w = opacity * max(1 - r^2, 0)^3.
// This deliberately does not use the beta kernel: the cubic has C2 support borders.
struct SharedHeightMember {
    float3 center{}, axisA{}, axisB{}, normal{}, dualA{}, dualB{}, slope{};
    float3 albedo{};
    float opacity = 0.0f;
    uint32_t index = kInvalidIndex;
};

struct SharedHeightSurface {
    static constexpr int capacity = 8;
    SharedHeightMember members[capacity];
    int memberCount = 2;
    float3 origin{}, normal{}, lower{}, upper{};
    bool valid = false;
};

inline bool prepareSharedHeightSurface(SharedHeightSurface &s) {
    s.valid = false;
    if (s.memberCount < 1 || s.memberCount > SharedHeightSurface::capacity) return false;
    for (int i = 0; i < s.memberCount; ++i) {
        auto &m = s.members[i];
        const float aa = dot(m.axisA, m.axisA), ab = dot(m.axisA, m.axisB);
        const float bb = dot(m.axisB, m.axisB), det = aa * bb - ab * ab;
        if (!(aa > 0.0f && bb > 0.0f && det > 1.0e-8f * aa * bb)) return false;
        m.normal = normalize(cross(m.axisA, m.axisB));
        m.dualA = (bb * m.axisA - ab * m.axisB) / det;
        m.dualB = (aa * m.axisB - ab * m.axisA) / det;
    }
    s.normal = float3{0.0f};
    s.origin = float3{0.0f};
    for (int i = 0; i < s.memberCount; ++i) {
        auto n = s.members[i].normal;
        if (dot(s.members[0].normal, n) < 0.0f) n = -n;
        s.normal += n;
        s.origin += s.members[i].center;
    }
    s.normal = normalize(s.normal);
    s.origin = s.origin / float(s.memberCount);
    s.lower = float3{INFINITY};
    s.upper = float3{-INFINITY};
    for (int i = 0; i < s.memberCount; ++i) {
        auto &m = s.members[i];
        const float denominator = dot(m.normal, s.normal);
        if (sycl::fabs(denominator) < 0.1f) return false;
        // Tangential gradient of this plane's height in the common chart.
        m.slope = s.normal - m.normal / denominator;
        const auto r2 = m.axisA * m.axisA + m.axisB * m.axisB;
        const float3 extent{sycl::sqrt(r2.x()), sycl::sqrt(r2.y()), sycl::sqrt(r2.z())};
        s.lower = min(s.lower, m.center - extent);
        s.upper = max(s.upper, m.center + extent);
    }
    // The reconstructed point is a convex combination of points on the ellipses.
    // Their joint world AABB therefore bounds the entire reconstructed surface.
    const float pad = 1.0e-6f * sycl::fmax(1.0f, length(s.upper - s.lower));
    s.lower -= float3{pad};
    s.upper += float3{pad};
    s.valid = true;
    return true;
}

struct SharedHeightEvaluation {
    float height = 0.0f;
    float3 gradient{}, albedo{};
    float opacity = 0.0f;
    bool valid = false;
};

inline SharedHeightEvaluation evaluateSharedHeight(const SharedHeightSurface &s, const float3 &x) {
    SharedHeightEvaluation e;
    const float3 q = x - s.normal * dot(s.normal, x - s.origin);
    double sum = 0.0, heights = 0.0;
    float3 gradientW{0.0f}, gradientH{0.0f}, color{0.0f};
    float transmission = 1.0f;
    for (int i = 0; i < s.memberCount; ++i) {
        const auto &m = s.members[i];
        const float z = dot(m.normal, m.center - q) / dot(m.normal, s.normal);
        const float3 local = q + s.normal * z - m.center;
        const float u = dot(m.dualA, local), v = dot(m.dualB, local);
        const float base = 1.0f - u * u - v * v;
        if (base <= 0.0f || m.opacity <= 0.0f) continue;
        const float w = m.opacity * base * base * base;
        const float3 du = m.dualA - s.normal * dot(m.dualA, s.normal) +
                          m.slope * dot(m.dualA, s.normal);
        const float3 dv = m.dualB - s.normal * dot(m.dualB, s.normal) +
                          m.slope * dot(m.dualB, s.normal);
        const float3 dw = -6.0f * m.opacity * base * base * (u * du + v * dv);
        sum += w;
        heights += double(w) * z;
        gradientW += dw;
        gradientH += dw * z + w * m.slope;
        color += w * m.albedo;
        transmission *= 1.0f - w;
    }
    if (!(sum > 1.0e-30)) return e;
    e.height = float(heights / sum);
    e.gradient = (gradientH - e.height * gradientW) / float(sum);
    e.albedo = color / float(sum);
    e.opacity = 1.0f - transmission;
    e.valid = true;
    return e;
}

inline double sharedHeightPolynomialValue(const double *p, int degree, double x) {
    double value = p[degree];
    for (int i = degree - 1; i >= 0; --i) value = value * x + p[i];
    return value;
}

// Isolate all real roots in [0,1] by derivative roots, then bisect each monotone
// interval. No recursion (device-safe). Double precision limits cancellation in
// the degree-seven polynomial near compact-support boundaries.
inline int sharedHeightPolynomialRoots(const double *p, double *roots) {
    double derivatives[8][8]{};
    double previous[8]{}, current[8]{};
    int degree = 7;
    while (degree > 0 && p[degree] == 0.0) --degree;
    if (degree == 0) return 0; // A ray contained in the surface has no isolated hit.
    for (int i = 0; i <= degree; ++i) derivatives[degree][i] = p[i];
    for (int d = degree - 1; d >= 1; --d)
        for (int i = 0; i <= d; ++i)
            derivatives[d][i] = (i + 1) * derivatives[d + 1][i + 1];
    int count = 0;
    for (int d = 1; d <= degree; ++d) {
        int nextCount = 0;
        double scale = 0.0;
        for (int i = 0; i <= d; ++i) scale += sycl::fabs(derivatives[d][i]);
        const double tolerance = 1.0e-13 * scale;
        double left = 0.0;
        double fl = sharedHeightPolynomialValue(derivatives[d], d, left);
        if (sycl::fabs(fl) <= tolerance) current[nextCount++] = left;
        for (int interval = 0; interval <= count; ++interval) {
            const double right = interval < count ? previous[interval] : 1.0;
            const double fr = sharedHeightPolynomialValue(derivatives[d], d, right);
            if ((fl < -tolerance && fr > tolerance) || (fl > tolerance && fr < -tolerance)) {
                double a = left, b = right, fa = fl;
                for (int step = 0; step < 44; ++step) {
                    const double mid = 0.5 * (a + b);
                    const double fm = sharedHeightPolynomialValue(derivatives[d], d, mid);
                    if ((fa < 0.0) == (fm < 0.0)) { a = mid; fa = fm; }
                    else b = mid;
                }
                if (nextCount < 8) current[nextCount++] = 0.5 * (a + b);
            }
            if (sycl::fabs(fr) <= tolerance && nextCount < 8 &&
                (nextCount == 0 || right - current[nextCount - 1] > 1.0e-10))
                current[nextCount++] = right;
            left = right;
            fl = fr;
        }
        count = nextCount;
        for (int i = 0; i < count; ++i) previous[i] = current[i];
    }
    for (int i = 0; i < count; ++i) roots[i] = previous[i];
    return count;
}

inline bool intersectSharedHeight(const SharedHeightSurface &s, const Ray &ray,
                                 float tMin, float tMax, float &tHit,
                                 SharedHeightEvaluation &evaluation) {
    // This degree-seven solver is the original pair experiment. Multi-member
    // charts use intersectSharedSlabs, which includes all chart contributors.
    if (!s.valid || s.memberCount != 2) return false;
    double near = tMin, far = tMax;
    for (int axis = 0; axis < 3; ++axis) {
        if (sycl::fabs(ray.direction[axis]) < 1.0e-20f) {
            if (ray.origin[axis] < s.lower[axis] || ray.origin[axis] > s.upper[axis]) return false;
        } else {
            double a = (double(s.lower[axis]) - ray.origin[axis]) / ray.direction[axis];
            double b = (double(s.upper[axis]) - ray.origin[axis]) / ray.direction[axis];
            near = sycl::fmax(near, sycl::fmin(a, b));
            far = sycl::fmin(far, sycl::fmax(a, b));
        }
    }
    if (!(far > near)) return false;
    // Work near the object rather than subtracting large camera-ray distances.
    const float3 start = ray.origin + float(near) * ray.direction;
    const float3 delta = float(far - near) * ray.direction;
    double u[2]{}, v[2]{}, du[2]{}, dv[2]{}, residual[2]{}, dr[2]{};
    double events[6]{0.0, 1.0};
    int eventCount = 2;
    for (int i = 0; i < 2; ++i) {
        const auto &m = s.members[i];
        const float denom = dot(m.normal, s.normal);
        residual[i] = dot(m.normal, start - m.center) / denom;
        dr[i] = dot(m.normal, delta) / denom;
        const float3 p = start - float(residual[i]) * s.normal - m.center;
        const float3 dp = delta - float(dr[i]) * s.normal;
        u[i] = dot(m.dualA, p); v[i] = dot(m.dualB, p);
        du[i] = dot(m.dualA, dp); dv[i] = dot(m.dualB, dp);
        const double a = du[i] * du[i] + dv[i] * dv[i];
        const double b = u[i] * du[i] + v[i] * dv[i];
        const double c = u[i] * u[i] + v[i] * v[i] - 1.0;
        const double disc = b * b - a * c;
        if (a > 0.0 && disc > 0.0) {
            // Stable quadratic roots, including narrow projected footprints.
            const double q = -b - sycl::copysign(sycl::sqrt(disc), b);
            const double r0 = q / a, r1 = q != 0.0 ? c / q : -b / a;
            if (r0 > 0.0 && r0 < 1.0) events[eventCount++] = r0;
            if (r1 > 0.0 && r1 < 1.0) events[eventCount++] = r1;
        }
    }
    for (int i = 1; i < eventCount; ++i) {
        const double value = events[i]; int j = i;
        while (j > 0 && events[j - 1] > value) { events[j] = events[j - 1]; --j; }
        events[j] = value;
    }
    for (int segment = 0; segment + 1 < eventCount; ++segment) {
        const double a = events[segment], width = events[segment + 1] - a;
        if (width <= 1.0e-14) continue;
        double polynomial[8]{};
        for (int i = 0; i < 2; ++i) {
            const double um = u[i] + du[i] * (a + width * 0.5);
            const double vm = v[i] + dv[i] * (a + width * 0.5);
            if (um * um + vm * vm >= 1.0 || s.members[i].opacity <= 0.0f) continue;
            const double ua = u[i] + du[i] * a, va = v[i] + dv[i] * a;
            const double ud = du[i] * width, vd = dv[i] * width;
            const double base[3]{1.0 - ua * ua - va * va,
                                 -2.0 * (ua * ud + va * vd), -ud * ud - vd * vd};
            double cube[7]{};
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l) cube[j + k + l] += base[j] * base[k] * base[l];
            for (int j = 0; j < 7; ++j) {
                polynomial[j] += s.members[i].opacity * cube[j] * (residual[i] + a * dr[i]);
                polynomial[j + 1] += s.members[i].opacity * cube[j] * width * dr[i];
            }
        }
        double roots[8]{};
        const int rootCount = sharedHeightPolynomialRoots(polynomial, roots);
        for (int r = 0; r < rootCount; ++r) {
            const float t = float(near + (far - near) * (a + width * roots[r]));
            if (t <= tMin || t >= tMax) continue;
            const float3 x = ray.origin + t * ray.direction;
            const auto e = evaluateSharedHeight(s, x);
            // Reject zeros caused only by vanishing support, and ill-conditioned
            // polynomial roots. There is no fallback to constituent plane hits.
            const float tolerance = 2.0e-5f * sycl::fmax(1.0e-3f, length(s.upper - s.lower));
            if (e.valid && sycl::fabs(dot(s.normal, x - s.origin) - e.height) <= tolerance) {
                tHit = t; evaluation = e; return true;
            }
        }
    }
    return false;
}

} // namespace Pale
