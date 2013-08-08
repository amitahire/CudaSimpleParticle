#pragma once

#include "Mesh.h"
#define RAW_WIDTH 257
#define RAW_HEIGHT 257

#define WATERMAP_X 16.0f
#define WATERMAP_Z 16.0f
#define WATERMAP_Y 1.25f
#define WATERMAP_TEX_X 1.0f / 16.0f
#define WATERMAP_TEX_Z 1.0f / 16.0f

class WaterMap : public Mesh {
public :
	WaterMap();
	~ WaterMap ( void ){};
};