#include "ParticleSystem.h"
ParticleSystem* ParticleSystem::instance = 0;

void ParticleSystem::resizeBuffer(int size) {
	
	//Delete old things not needed.
	delete[] vertices;
	delete[] colours;
	glDeleteBuffers(1, &bufferObject[VERTEX_BUFFER]);
	glDeleteBuffers(1, &bufferObject[COLOUR_BUFFER]);

	// Regenerate Buffers.
	vertices = new Vector3[size];
	colours  = new Vector4[size];


	glGenBuffers(1, &bufferObject[VERTEX_BUFFER]);
	glBindBuffer(GL_ARRAY_BUFFER, bufferObject[VERTEX_BUFFER]);
	glBufferData(GL_ARRAY_BUFFER, size*sizeof(Vector3), 0, GL_DYNAMIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, size*sizeof(Vector3), 0);


	glGenBuffers(1, &bufferObject[COLOUR_BUFFER]);
	glBindBuffer(GL_ARRAY_BUFFER, bufferObject[COLOUR_BUFFER]);
	glBufferData(GL_ARRAY_BUFFER, size*sizeof(Vector4), 0, GL_DYNAMIC_DRAW);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ParticleSystem::Draw()	{

	glBindVertexArray(arrayObject);

	// Bind the Buffers.
	glBindBuffer(GL_ARRAY_BUFFER, bufferObject[VERTEX_BUFFER]);
	glVertexAttribPointer(VERTEX_BUFFER, 3, GL_FLOAT, GL_FALSE, sizeof(Vector3), 0);
	glEnableVertexAttribArray(VERTEX_BUFFER);
	

	glBindBuffer(GL_ARRAY_BUFFER, bufferObject[COLOUR_BUFFER]);
	glVertexAttribPointer(COLOUR_BUFFER, 4, GL_FLOAT, GL_FALSE, sizeof(Vector4), 0);
	glEnableVertexAttribArray(COLOUR_BUFFER);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE,GL_ONE);
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	if(bufferObject[INDEX_BUFFER]) {
		glDrawElements(type, numIndices, GL_UNSIGNED_INT, 0);
	} else {
		glDrawArrays(GL_POINTS,  0, bufferSize);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindVertexArray(0);
};
