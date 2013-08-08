#version 150 core

uniform mat4 modelMatrix ;
uniform mat4 viewMatrix ;
uniform mat4 projMatrix ;

in vec3 position;
in vec4 colour;

out Vertex{
	vec4 colour;
	vec3 worldPos ;
}OUT;

void main(void){

	OUT . worldPos = ( modelMatrix * vec4 ( position ,1)). xyz ;
	gl_Position = ( projMatrix * viewMatrix * modelMatrix ) * vec4 ( position , 1.0);
	
	OUT.colour = colour;
}