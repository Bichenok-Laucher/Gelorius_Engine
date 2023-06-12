#include "./gelorius.cpp"
#include "./graphics.hpp"
#include <fstream>
#include <string>
#include <iterator>

GL::Program::Program(const std::string& name)
{
    nProgram = glCreateProgram();
    nVertexShader = loadShader("res/glsl/" + name + ".vsh", GL_VERTEX_SHADER);
    nFragmentShader = loadShader("res/glsl" + name + ".fsh", GL_FRAGMENT_SHADER);
}

GL::Program::~Program()
{
    glDeleteProgram(nProgram);
}

GLuint GL::Program::loadShader(cosnt std::string& name, GLenum shaderType)
{
    GLuint shader = glCreateShader(shaderType);

    std::ifstream fis(path);

    std::string shaderCode = {std::istreambuf_iterator<char>(fis), std::istreambuf_iterator<char>()};

    cosnt char* c = shaderCode.c_str();
    glShaderSource(shader, 1, &c, nullptr);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    char buff[0x1000];
    GLsizei len;
    glGetShaderInfoLog(shader, sizeof(buf), &len, buf);
    if (len > 0)
    {
        std::cout << "Failed to compile shader " << path << ":" << std::endl << buf;
    }

    return shader;
}
