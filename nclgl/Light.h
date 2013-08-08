#pragma once
#include "Vector4.h"
#include "Vector3.h"
class Light {
public :
	Light () {
		this -> position = Vector3(0.0f,0.0f,0.0f) ;
		this -> light_colour = Vector4(1.0,1.0,1.0,1.0);
		this -> radius = 100 ;
		this -> direction = Vector3(0,0,0);
		this -> spotLight = false;
		this -> laserLight = false; 
	}
	Light ( Vector3 position , Vector4 light_colour , float radius ) {
		this -> position = position ;
		this -> light_colour = light_colour ;
		this -> radius = radius ;
		this -> spotLight = false;
		this -> laserLight = false; 
	}

	Light ( Vector3 position , Vector4 light_colour , float radius ,Vector3 direction) {
		this -> position = position ;
		this -> light_colour = light_colour ;
		this -> radius = radius ;
		this -> direction = direction ;
		this -> spotLight = true;
		this -> laserLight = false; 
	}

	~ Light ( void ){};

	Vector3 GetPosition () const { return position ; }
	void SetPosition ( Vector3 val ) { position = val ; }

	float GetRadius () const { return radius ; }
	void SetRadius ( float val ) { radius = val ; }

	Vector4 GetColour () const { return light_colour ; }
	void SetColour ( Vector4 val ) { light_colour = val ; }

	Vector3 GetDirection () const { return direction ; }
	void SetDirection ( Vector3 val ) { direction = val ; }

	void SetIsSpotLight(bool isSpot){spotLight = isSpot;}
	bool isSpotLight() const { return spotLight ; }
	
	void SetIsLaserLight(bool isLaser){laserLight = isLaser;}
	bool isLaserLight() const { return laserLight ; }

protected :
	bool laserLight;
	bool spotLight;
	Vector3 direction;
	Vector3 position ;
	Vector4 light_colour ;
	float radius ;
};