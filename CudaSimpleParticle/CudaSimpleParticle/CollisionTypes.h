#pragma once
#include "../../nclgl/Vector3.h"

//Collision Shape Base Class
class CollisionShape {	
public:
	Vector3 pos;
	//Used For One to many Collision
	void* ListPtr;
};

//Collision Shape
class CollisionSphere :public CollisionShape{
public :
	CollisionSphere ( const Vector3 & p , float r ){

		pos = p;
		radius = r;
	}
	float radius;
};

class CollisionAABB :public CollisionShape{
public :
	CollisionAABB(){}; // This might not be required. Clean it up later. 
	CollisionAABB ( const Vector3 &p , const Vector3 &hd ){

		pos = p ;
		halfdims = hd ;
	}
	Vector3 halfdims ;
};

//Collision Result
class CollisionData {	
public :
	Vector3 m_point ;
	Vector3 m_normal ;
	float m_penetration ;
};
