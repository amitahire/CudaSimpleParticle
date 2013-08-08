#pragma once
#include "SceneNode.h"
#include "Mesh.h"
#include "Light.h"

class LightNode: public SceneNode{
public:
	LightNode(Mesh *m = NULL , Vector4 colour = Vector4 (1 ,1 ,1 ,1));
	~LightNode(void);

	virtual void Draw (const OGLRenderer &r);
	Light* nodeLight;	
	bool drawSelf;
};