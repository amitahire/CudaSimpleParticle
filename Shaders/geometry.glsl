#version 150

uniform float particleSize;

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in Vertex {
	vec4 colour;
} IN[];

out Vertex {
	vec4 colour;
	vec2 texCoord;
} OUT;
 
void main() {	

	for(int i = 0; i < gl_in.length(); ++i) {

		OUT.colour = IN[i].colour;

		//But hang on! We don't put positions in our OUT struct!
		//That's ok, we can get positions using another array -
		//gl_in also works as an array, containing gl_Positions.
		//these are the outputted vertex positions created by
		//our vertex shader.

		//Now we have a position for the input vertex, we can 
		//set the output vertex position and texture coordinate.

		//We have some slightly odd syntax, in that we always
		//put the position data in (gl_Position), but data we
		//want sent to the fragment shader gets put in the OUT
		//struct...

		//top right
		gl_Position = gl_in[ i ].gl_Position;
		gl_Position.x += particleSize;
		gl_Position.y += particleSize;
		OUT.texCoord = vec2(1,0);

		//When we've created a vertex, we call the glsl in-built
		//function EmitVertex, which sends our vertex off for
		//rasterisation.
		EmitVertex();

		//Then we do the other vertices of the quad...
		//Top Left
		gl_Position = gl_in[ i ].gl_Position;
		gl_Position.x -= particleSize;
		gl_Position.y += particleSize;
		OUT.texCoord = vec2(0,0);
		EmitVertex();

		//bottom right
		gl_Position = gl_in[ i ].gl_Position;
		gl_Position.x += particleSize;
		gl_Position.y -= particleSize;
		OUT.texCoord = vec2(1,1);
		EmitVertex();

		//bottom Left
		gl_Position = gl_in[ i ].gl_Position;
		gl_Position.x -= particleSize;
		gl_Position.y -= particleSize;
		OUT.texCoord = vec2(0,1);
		EmitVertex();

		//Once we've emitted 4 vertices, we tell glsl that we've 
		//finished a complete primitive, so it can begin the 
		//rasterisation process on it.
		EndPrimitive();
	}
}

//That's pretty much everything for geometry shaders! If you've taken a look
//at the 'Drawing Text' tutorial, you might want to start thinking about how
//you could start using a geometry shader to render the quads used for every
//character of text. The Texture Atlas tutorial should also help in working
//out which texture coordinates to give to a vertex...