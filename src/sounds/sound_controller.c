#include "./glfw/include/GLFW/glfw3.h"
#include "../graphics.hpp"
#include "./glfw/include/GLFW/glfw3native.h"

int main() {

GL_DEVICE* device = GL_OPEN_DEVICE(nullptr);
GL_CONTEXT* context = GL_CREATE_CONTEXT(device, nullptr);
GL_MAKE_CONTEXT_CURRENT(context);

ALuint buffer;
GL_GEN_BUFFERS(1, &buffer);

ALuint source;
GL_GET_SOURCES(1, &source);

const char* GLFW_SOUND_FILE = gelorius_curent_sound_file;
    ALenum format;
    ALsizei size;
    ALvoid* data;
    ALsizei freq;
    ALboolean loop = AL_FALSE;

GL_LOAD_SOUND_FILE((ALByte*)soundFile, &format, &data, &size, &freq &loop);
GL_BUFFER_DATA(buffer, format, data, size, freq);
GL_UNLOAD_SOUND(format, data, size, freq);

gl_Sourcef(source, AL_GAIN, 1.0f);  // Set the volume (1.0 = maximum)
gl_Sourcef(source, AL_PITCH, 1.0f);  // Set the pitch (1.0 = normal)
gl_Sourcei(source, AL_LOOPING, AL_FALSE);
GL_SOURCE_PLAY(source);

ALint state;
do {
    GL_GET_SOURCEI(source, GL_SOURCE_STATE, &state);
}
 while (state == GL_PLAYING);

GL_DELETE_SOURCES(1, &source);
GL_DELETE_BUFFERS(1, &buffer);
GL_MAKE_CONTEXT_CURRENT(nullptr);
GL_DESTROY_CONTEXT(context);
GL_CLOSE_DEVICE(device);

    return 0;
}