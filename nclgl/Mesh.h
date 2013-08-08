#pragma once
#include "../../nclgl/OGLRenderer.h"
#include <iostream>
enum MeshBuffer {
	VERTEX_BUFFER,COLOUR_BUFFER,TEXTURE_BUFFER,NORMAL_BUFFER,TANGENT_BUFFER,INDEX_BUFFER,MAX_BUFFER
};

class Mesh {

public :
	//Constructor-Destructor
	Mesh ( void );
	~ Mesh ( void );
	virtual void Draw ();
	
	//Texture Getter-Setter
	void SetTexture(GLuint tex){ texture = tex;}
	GLuint GetTexture() {return texture;}
	
	//Indices Buff
	GLuint numIndices;
	unsigned int* indices;

	//Bump Map
	void SetBumpMap(GLuint tex) { bumpTexture = tex;}
	GLuint GetBumpMap() { return bumpTexture ;}

	//Shape Generation
	static Mesh * GenerateTriangle();	
	static Mesh * GenerateQuad();
	static Mesh * GenerateBox();
	static Mesh * GenerateM(float r,float g,float b,float a);

	GLuint GetVBOid() {return arrayObject;};

protected :
	//Mesh
	virtual void BufferData();
	GLuint arrayObject; 
	GLuint bufferObject[MAX_BUFFER];
	
	//Shape 
	GLuint type;
	GLuint numVertices;
	Vector3 * vertices;
	Vector4 * colours;

	//Texture
	GLuint texture;
	Vector2* textureCoords;

	// Normals
	void GenerateNormals();
	Vector3 * normal ;

	//Bump Map
	void GenerateTangents();
	Vector3 GenerateTangent( const Vector3 &a, const Vector3 &b,const Vector3 &c, const Vector2 &ta ,const Vector2 &tb , const Vector2 &tc );
	Vector3 *tangents;
	GLuint bumpTexture;

};
