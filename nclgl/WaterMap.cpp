#include "WaterMap.h"
#include <math.h>
#include <time.h>

WaterMap::WaterMap(){
	srand(0);

	numVertices = RAW_WIDTH*RAW_HEIGHT;
	numIndices = (RAW_WIDTH-1)*(RAW_HEIGHT-1)*6;
	vertices = new Vector3[numVertices];
	textureCoords = new Vector2[numVertices];
	indices = new GLuint[numIndices];
	
	float sinAngle = 0.0f;
	float sinHeight = sin(sinAngle);
	float cosHeight = cos(sinAngle);
	for (int x = 0; x < RAW_WIDTH ; ++ x ) {
		for (int z = 0; z < RAW_HEIGHT ; ++ z ) {
			sinHeight = sin(sinAngle*((double)(rand()%10)*0.01));	
			cosHeight = cos(z * WATERMAP_Z);
			//sinHeight = sin(((double)(rand()%361))*0.1);	
			//cosHeight = cos(((double)(rand()%361))*0.1);
			int offset = ( x * RAW_WIDTH ) + z ;
			vertices[offset] = Vector3(x*WATERMAP_X , (3*cosHeight*sinHeight*WATERMAP_Y) +60, z * WATERMAP_Z );
			textureCoords[offset] = Vector2(x*WATERMAP_TEX_X , z * WATERMAP_TEX_Z );
			sinAngle += 1.0f;
		}
	}

	numIndices = 0;
	for (int x = 0; x < RAW_WIDTH -1; ++ x ) {
		for (int z = 0; z < RAW_HEIGHT -1; ++ z ) {
			int a = ( x * ( RAW_WIDTH )) + z ;
			int b = (( x +1) * ( RAW_WIDTH )) + z ;
			int c = (( x +1) * ( RAW_WIDTH )) + ( z +1);
			int d = ( x * ( RAW_WIDTH )) + ( z +1);

			indices [numIndices ++] = c ;
			indices [numIndices ++] = b ;
			indices [numIndices ++] = a ;

			indices [numIndices ++] = a ;
			indices [numIndices ++] = d ;
			indices [numIndices ++] = c ;
		}
	}
	GenerateTangents();
	GenerateNormals();
	BufferData();
	srand((unsigned int)time(NULL));
}
