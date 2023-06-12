#include "./Vulkan/base/VulkanDebug.h"

class Camera
{
    private:
        float fov;
        float znear, zfar;

        void updateViewMatrix()
        {
            glm::mat4 rotM = glm::mat4(1.0f);
            glm::mat4 transM;

            rotM = glm::rotate(rotM, glm::radians(rotation.x * (flipY ? -1.0f : 1.0f)), glm::vec3(1.0f, 0.0f, 0.0f));
            rotM = glm::rotate(rotM, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
            rotM = glm::rotate(rotM, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

            glm::vec3 translation = position;
            if (flipY) {
                translation.y *= -1.0f;
            }
            transM = glm::translate(glm::mat4(1.0f), translation);

            if (type == CameraType::firstperson)
            {
                matrices.view = rotM * transM;
            }
            else
            {
                matrices.view = transM * rotM;
            }
            viewPos = glm::vec4(position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);
        }
}