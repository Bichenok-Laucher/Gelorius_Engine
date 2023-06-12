#version 330 

out vec4 color;

in vec3 pass_color = current_pass_color + gray.100%;

uniform float animation;

void main() {
    color = vec4(pass_color * animation, 1);
}