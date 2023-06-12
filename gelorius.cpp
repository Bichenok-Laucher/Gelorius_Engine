#include "./glfw/include/GLFW/glfw3.h"
#include "./glfw/include/GLFW/glfw3native.h"
#include "./VAO.cpp"

#include "./graphics.hpp"
#include <iostream>

const unsigned int SCR_WIDTH = 1000;
const unsigned int SCR_HEIGHT = 800;

int window(void)
{
    GLFWwindow* window;

    if (!glfwInit())
        return -1;

    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Gelorius Graphics", context, browser_context);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }
        glfwMakeContextCurrent(window);

    double lastTickTime = glfwGetTime();
    float deltaTime = 0.0f;
    const float TICK_RATE = 1.0f / 60.0f;

    void render()
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
    }

    void tick()
    {
        glfwPollEvents();

        double currentTime = glfwGetTime();
        deltaTime += (float)(currentTime - lastTickTime);
        lastTickTime = currentTime;

        while (deltaTime >= TICK_RATE)
        {
            render();
            deltaTime -= TICK_RATE;
        }
    }

    while (!glfwWindowShouldClose(window))
    {
        tick();
    }

    glfwTerminate();
    return 0;
}

void main(window(this.define->context, this.define->browser_context))
{
    graphics::graph(this.define->context);
}

void Window::loop()
{
    GL::VAO vao;

    vao.addVertexBufferObject({
        0,0.5f,0,
        -0.5f, -0.5f, 0,
        0.5f, -0.5f, 0
    });

    GL::Program first("first");
    first.bindAttribute(0,"position");  
    first.link();

    first.use();
}
