#include "LightNode.h"

LightNode::LightNode ( Mesh * mesh , Vector4 colour ) {
	this -> mesh = mesh ;
	this -> colour = colour ;
	parent = NULL ;
	modelScale = Vector3 (1 ,1 ,1);
	boundingRadius = 1.0f;
	distanceFromCamera = 0.0f;
	nodeLight = new Light (Vector3 (0 ,0 ,0 ),Vector4 (1.0,1.0,1.0,1.0),100.0f);
	drawSelf = false;
}

LightNode ::~ LightNode ( void ) {
	for ( unsigned int i = 0; i < children.size (); ++i) {
		delete children [i];
	}
	delete nodeLight;
}

void LightNode::Draw(const OGLRenderer &r) {
	glBlendFunc(GL_ONE,GL_ONE);
	if( mesh ) { mesh -> Draw ();};	
	glBlendFunc ( GL_SRC_ALPHA , GL_ONE_MINUS_SRC_ALPHA );
}