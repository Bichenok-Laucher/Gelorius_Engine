#include "./glfw/include/GLFW/glfw3.h"
#include "./glfw/include/GLFW/glfw3native.h"
#include "./json/include/json.hpp"
#include "./Vulkan/base/VulkanDebug.h"
#include "./camera.hpp"
#include "./RecordRenderDoc.h"
#include "./shaders.cpp"
#include "./src/sounds/sound_controller.c"
#include "./vertex_machine.cpp"

void GL::VAO::addVertexBufferObject(const std::vector<float>& data)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(nBuffers.size(), 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    nBuffers.push_back(vbo);
}