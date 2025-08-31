//
// Created by magnus on 8/29/25.
//

#include "Renderer/Kernels/PathTracerKernels.h"

namespace Pale {
    void PathTracerMeshKernel::traceOnePhoton(uint64_t photonID, uint32_t totalPhotonCount) const {
        m_sensor.framebuffer[photonID] = 0.3f;
    }
}
