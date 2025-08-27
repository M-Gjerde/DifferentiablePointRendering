import MG.DeviceSelector;
import MG.Scene;
import MG.SceneSerializer;

#include <memory>

int main() {
    // Select suitable SYCL device
    MG::DeviceSelector selector;

    // Load in xml file and Create Scene from xml
    std::shared_ptr<MG::Scene> scene = std::make_shared<MG::Scene>();
    MG::SceneSerializer serializer(scene);
    serializer.deserialize("../Scenes/cornell_box.xml");

    // Initialize a path tracer from sycl

    // Register scene in Path tracer
    // -- Create rendering primitives: BVH etc..

    // Render the image

    // Write output to file

    return 0;
}
