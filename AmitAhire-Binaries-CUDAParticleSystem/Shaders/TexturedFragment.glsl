#version 150 core

uniform sampler2D diffuseTex ;

in Vertex {
	vec2 texCoord;
} IN ;

out vec4 gl_FragColor;

void main ( void ) {

	vec4 colour =texture(diffuseTex,IN.texCoord);
	
	if((colour.a) == 0) {
		discard;
	}	

	gl_FragColor = colour;
    //gl_FragColor = vec4(IN.texCoord, 0, 1);
	//gl_FragColor = IN.colour;
	
}