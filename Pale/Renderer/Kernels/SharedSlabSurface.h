#pragma once

#include "SharedHeightSurface.h"

namespace Pale {

struct SharedSlabChart {
    SharedHeightSurface height;
    float3 tangent{};
    float halfWidth = 1.25f, halfDepth = 0.6f;
    float lateralShift = 0.0f, depthShift = 0.0f;
};

struct SharedSlabSurface {
    SharedSlabChart charts[2];
    float3 lower{}, upper{};
    float coverage = 6.0f;
    bool valid = false;
};

inline bool prepareSharedSlabSurface(SharedSlabSurface &s) {
    s.valid = false;
    s.lower = float3{INFINITY}; s.upper = float3{-INFINITY};
    for (int k = 0; k < 2; ++k) {
        auto &chart = s.charts[k]; auto &h = chart.height;
        if (!(chart.halfWidth > 0.0f && chart.halfDepth > 0.0f) || !prepareSharedHeightSurface(h)) return false;
        // One connected sheet needs a consistent signed-field orientation.
        if (k == 1 && dot(h.normal, s.charts[0].height.normal) < 0.0f) {
            h.normal = -h.normal;
            for (int i = 0; i < h.memberCount; ++i) h.members[i].slope = -h.members[i].slope;
        }
        if (k == 1 && dot(h.normal, s.charts[0].height.normal) < 0.1f) return false;
        chart.tangent = normalize(h.members[0].axisA - h.normal * dot(h.normal, h.members[0].axisA));
        const auto v = cross(h.normal, chart.tangent);
        float vExtent = 0.0f;
        for (int i = 0; i < h.memberCount; ++i) {
            const auto &m = h.members[i];
            const float a = dot(v, m.axisA), b = dot(v, m.axisB);
            vExtent = sycl::fmax(vExtent, sycl::fabs(dot(v, m.center - h.origin)) + sycl::sqrt(a * a + b * b));
        }
        // Bound the slabs themselves. The ellipse-union bound for a single
        // chart does not generally bound a blend of differently oriented charts.
        const auto center = h.origin + chart.lateralShift * chart.tangent + chart.depthShift * h.normal;
        float3 extent;
        for (int axis = 0; axis < 3; ++axis)
            extent[axis] = sycl::fabs(chart.tangent[axis]) * chart.halfWidth +
                sycl::fabs(v[axis]) * vExtent + sycl::fabs(h.normal[axis]) * chart.halfDepth + 1.0e-6f;
        s.lower = min(s.lower, center - extent); s.upper = max(s.upper, center + extent);
    }
    s.valid = true;
    return true;
}

struct SharedSlabEvaluation {
    float value = 0.0f;
    float3 gradient{}, albedo{};
    float opacity = 0.0f, weightB = 0.0f;
    bool valid = false;
};

inline SharedSlabEvaluation evaluateSharedSlabs(const SharedSlabSurface &s, const float3 &x) {
    SharedSlabEvaluation e;
    float total = 0.0f, numerator = 0.0f, weightB = 0.0f;
    float3 gradTotal{0.0f}, gradNumerator{0.0f}, color{0.0f};
    for (int k = 0; k < 2; ++k) {
        const auto &c = s.charts[k]; const auto &h = c.height;
        const float u = (dot(c.tangent, x - h.origin) - c.lateralShift) / c.halfWidth;
        const float z = (dot(h.normal, x - h.origin) - c.depthShift) / c.halfDepth;
        const float l = 1.0f - u * u, d = 1.0f - z * z;
        if (l <= 0.0f || d <= 0.0f) continue;
        const float band = l * l * l * d * d * d;
        const auto gradBand = (-6.0f * u * l * l * d * d * d / c.halfWidth) * c.tangent +
                              (-6.0f * z * d * d * l * l * l / c.halfDepth) * h.normal;
        for (int i = 0; i < h.memberCount; ++i) {
            const auto &m = h.members[i];
            const auto residualGradient = m.normal / dot(m.normal, h.normal);
            const float residual = dot(residualGradient, x - m.center);
            const auto local = x - residual * h.normal - m.center;
            const float a = dot(m.dualA, local), b = dot(m.dualB, local);
            const float base = 1.0f - a * a - b * b;
            if (base <= 0.0f || m.opacity <= 0.0f) continue;
            const float w = m.opacity * base * base * base;
            const auto da = m.dualA - dot(m.dualA, h.normal) * residualGradient;
            const auto db = m.dualB - dot(m.dualB, h.normal) * residualGradient;
            const auto dw = -6.0f * m.opacity * base * base * (a * da + b * db);
            const float weight = band * w;
            const auto gradient = band * dw + w * gradBand;
            total += weight;
            numerator += weight * residual;
            gradTotal += gradient;
            gradNumerator += gradient * residual + weight * residualGradient;
            color += weight * m.albedo;
            if (k == 1) weightB += weight;
        }
    }
    // This is sum(B_k F_k)/sum(B_k) with B_k = band_k * sum(w_ki).
    // Expanding the numerator avoids evaluating undefined heights at empty support.
    if (!(total > 1.0e-25f)) return e;
    e.value = numerator / total;
    e.gradient = (gradNumerator - e.value * gradTotal) / total;
    e.albedo = color / total;
    e.weightB = weightB / total;
    // Coverage is applied ONCE at the common hit; it fades at empty support.
    // The scale is explicit because chart density affects this experiment's alpha.
    e.opacity = -sycl::expm1(-s.coverage * total);
    e.valid = true;
    return e;
}

// Within fixed support, band*w*planeResidual has degree 6+6+6+1 = 19.
// Bernstein coefficients keep products of nonnegative footprint polynomials
// well-conditioned, unlike conversion of this high degree to a power basis.
constexpr int sharedSlabDegree = 19;
inline void slabMultiplyQuadratic(double *p, int degree, const double *q) {
    double out[20]{};
    const double denominator = double(degree + 2) * (degree + 1);
    for (int k = 0; k <= degree + 2; ++k) {
        if (k <= degree) out[k] += p[k] * q[0] * (degree + 2 - k) * (degree + 1 - k) / denominator;
        if (k > 0 && k <= degree + 1) out[k] += p[k - 1] * q[1] * 2 * k * (degree + 2 - k) / denominator;
        if (k > 1) out[k] += p[k - 2] * q[2] * k * (k - 1) / denominator;
    }
    for (int i = 0; i <= degree + 2; ++i) p[i] = out[i];
}

inline void slabBaseCoefficients(double u, double v, double du, double dv, double a, double b, double *out) {
    const double ua = u + a * du, va = v + a * dv;
    const double ub = u + b * du, vb = v + b * dv;
    out[0] = sycl::fmax(0.0, 1.0 - ua * ua - va * va);
    out[1] = 1.0 - ua * ub - va * vb;
    out[2] = sycl::fmax(0.0, 1.0 - ub * ub - vb * vb);
}

struct SharedSlabRayMember {
    double u, v, du, dv, residual, dr;
};
struct SharedSlabRayChart {
    double u, du, z, dz;
    SharedSlabRayMember members[SharedHeightSurface::capacity];
};

inline void slabAddSupportEvents(double u, double v, double du, double dv, double *events, int &count) {
    const double a = du * du + dv * dv, b = u * du + v * dv;
    const double c = u * u + v * v - 1.0, disc = b * b - a * c;
    if (!(a > 0.0 && disc > 0.0)) return;
    const double q = -b - sycl::copysign(sycl::sqrt(disc), b);
    const double roots[2]{q / a, q != 0.0 ? c / q : -b / a};
    for (double r : roots) if (r > 0.0 && r < 1.0) events[count++] = r;
}

inline bool slabAcceptRoot(const SharedSlabSurface &s, const Ray &ray, double t,
                          float tMin, float tMax, float &tHit, SharedSlabEvaluation &out) {
    if (!(t > tMin && t < tMax)) return false;
    const auto e = evaluateSharedSlabs(s, ray.origin + float(t) * ray.direction);
    const float tolerance = 3.0e-6f * sycl::fmax(1.0e-3f, length(s.upper - s.lower));
    if (!e.valid || sycl::fabs(e.value) > tolerance || dot(e.gradient, e.gradient) < 1.0e-12f) return false;
    tHit = float(t); out = e; return true;
}

inline bool slabFindRoot(const double *polynomial, const SharedSlabSurface &s, const Ray &ray,
                        double near, double far, float tMin, float tMax, float &tHit, SharedSlabEvaluation &out) {
    struct Node { double p[20], a, b; int depth; };
    Node stack[30]; int count = 1;
    for (int i = 0; i < 20; ++i) stack[0].p[i] = polynomial[i];
    stack[0].a = near; stack[0].b = far; stack[0].depth = 0;
    while (count > 0) {
        const Node node = stack[--count];
        double minimum = node.p[0], maximum = node.p[0];
        for (int i = 1; i < 20; ++i) {
            minimum = sycl::fmin(minimum, node.p[i]); maximum = sycl::fmax(maximum, node.p[i]);
        }
        // The convex hull of Bernstein coefficients bounds the polynomial.
        if (minimum > 0.0 || maximum < 0.0 || (minimum == 0.0 && maximum == 0.0)) continue;
        if (node.p[0] == 0.0 && slabAcceptRoot(s, ray, node.a, tMin, tMax, tHit, out)) return true;
        if (node.depth >= 27 || node.b - node.a < 1.0e-7 * sycl::fmax(1.0e-3f, length(s.upper - s.lower))) {
            if (slabAcceptRoot(s, ray, 0.5 * (node.a + node.b), tMin, tMax, tHit, out)) return true;
            continue;
        }
        if (minimum == 0.0 || maximum == 0.0) {
            // No interior root if all coefficients have the same weak sign.
            if (node.p[19] == 0.0 && slabAcceptRoot(s, ray, node.b, tMin, tMax, tHit, out)) return true;
            continue;
        }
        double triangle[20], left[20], right[20];
        for (int i = 0; i < 20; ++i) triangle[i] = node.p[i];
        left[0] = triangle[0]; right[19] = triangle[19];
        for (int level = 1; level < 20; ++level) {
            for (int i = 0; i < 20 - level; ++i) triangle[i] = 0.5 * (triangle[i] + triangle[i + 1]);
            left[level] = triangle[0]; right[19 - level] = triangle[19 - level];
        }
        const double mid = 0.5 * (node.a + node.b);
        // Push far first: depth-first traversal always resolves the nearest root.
        for (int side = 1; side >= 0; --side) {
            auto &child = stack[count++];
            for (int i = 0; i < 20; ++i) child.p[i] = side == 0 ? left[i] : right[i];
            child.a = side == 0 ? node.a : mid; child.b = side == 0 ? mid : node.b;
            child.depth = node.depth + 1;
        }
    }
    return false;
}

inline bool intersectSharedSlabs(const SharedSlabSurface &s, const Ray &ray, float tMin, float tMax,
                                 float &tHit, SharedSlabEvaluation &out) {
    if (!s.valid) return false;
    double near = tMin, far = tMax;
    for (int axis = 0; axis < 3; ++axis) {
        if (sycl::fabs(ray.direction[axis]) < 1.0e-20f) {
            if (ray.origin[axis] < s.lower[axis] || ray.origin[axis] > s.upper[axis]) return false;
        } else {
            const double a = (double(s.lower[axis]) - ray.origin[axis]) / ray.direction[axis];
            const double b = (double(s.upper[axis]) - ray.origin[axis]) / ray.direction[axis];
            near = sycl::fmax(near, sycl::fmin(a, b)); far = sycl::fmin(far, sycl::fmax(a, b));
        }
    }
    if (!(far > near)) return false;
    const auto start = ray.origin + float(near) * ray.direction;
    const auto delta = float(far - near) * ray.direction;
    SharedSlabRayChart charts[2];
    double events[2 + 4 * SharedHeightSurface::capacity + 8]{0.0, 1.0}; int eventCount = 2;
    for (int k = 0; k < 2; ++k) {
        const auto &c = s.charts[k]; const auto &h = c.height; auto &r = charts[k];
        r.u = (dot(c.tangent, start - h.origin) - c.lateralShift) / c.halfWidth;
        r.du = dot(c.tangent, delta) / c.halfWidth;
        r.z = (dot(h.normal, start - h.origin) - c.depthShift) / c.halfDepth;
        r.dz = dot(h.normal, delta) / c.halfDepth;
        slabAddSupportEvents(r.u, 0, r.du, 0, events, eventCount);
        slabAddSupportEvents(r.z, 0, r.dz, 0, events, eventCount);
        for (int i = 0; i < h.memberCount; ++i) {
            const auto &m = h.members[i]; auto &p = r.members[i];
            const auto normal = m.normal / dot(m.normal, h.normal);
            p.residual = dot(normal, start - m.center); p.dr = dot(normal, delta);
            const auto local = start - float(p.residual) * h.normal - m.center;
            const auto direction = delta - float(p.dr) * h.normal;
            p.u = dot(m.dualA, local); p.v = dot(m.dualB, local);
            p.du = dot(m.dualA, direction); p.dv = dot(m.dualB, direction);
            slabAddSupportEvents(p.u, p.v, p.du, p.dv, events, eventCount);
        }
    }
    for (int i = 1; i < eventCount; ++i) {
        const double v = events[i]; int j = i;
        while (j > 0 && events[j - 1] > v) { events[j] = events[j - 1]; --j; }
        events[j] = v;
    }
    for (int segment = 0; segment + 1 < eventCount; ++segment) {
        const double a = events[segment], b = events[segment + 1], mid = 0.5 * (a + b);
        if (b - a < 1.0e-14) continue;
        double polynomial[20]{};
        for (int k = 0; k < 2; ++k) {
            const auto &r = charts[k]; const auto &h = s.charts[k].height;
            if (sycl::fabs(r.u + mid * r.du) >= 1.0 || sycl::fabs(r.z + mid * r.dz) >= 1.0) continue;
            double lateral[3], depth[3];
            slabBaseCoefficients(r.u, 0, r.du, 0, a, b, lateral);
            slabBaseCoefficients(r.z, 0, r.dz, 0, a, b, depth);
            for (int i = 0; i < h.memberCount; ++i) {
                const auto &p = r.members[i];
                const double u = p.u + mid * p.du, v = p.v + mid * p.dv;
                if (u * u + v * v >= 1.0 || h.members[i].opacity <= 0.0f) continue;
                double footprint[3]; slabBaseCoefficients(p.u, p.v, p.du, p.dv, a, b, footprint);
                double weight[20]{double(h.members[i].opacity)}; int degree = 0;
                for (int power = 0; power < 3; ++power) {
                    slabMultiplyQuadratic(weight, degree, footprint); degree += 2;
                    slabMultiplyQuadratic(weight, degree, lateral); degree += 2;
                    slabMultiplyQuadratic(weight, degree, depth); degree += 2;
                }
                const double f0 = p.residual + a * p.dr, f1 = p.residual + b * p.dr;
                for (int j = 0; j < 20; ++j) {
                    if (j < 19) polynomial[j] += double(19 - j) / 19.0 * weight[j] * f0;
                    if (j > 0) polynomial[j] += double(j) / 19.0 * weight[j - 1] * f1;
                }
            }
        }
        if (slabFindRoot(polynomial, s, ray, near + (far - near) * a, near + (far - near) * b,
                         tMin, tMax, tHit, out)) return true;
    }
    return false;
}

} // namespace Pale
