#include "HeightMap.h"
//#include "../CudaSimpleParticle/CudaSimpleParticle/PerlinNoise.h"
#include <math.h>
#include <time.h>

#define usePerlinNoise true

HeightMap::HeightMap(std::string name){
	//Temp Variable
	Vector3 maxXYZ = Vector3(0);
	Vector3 minXYZ = Vector3(0);

	////Setup Perlin noise Generater
	//srand((unsigned int) time(NULL)); //Reseed Random Seed....
	//int randomSeed = rand();
	//PerlinNoise* perlinNoiseF1 = new PerlinNoise(0.45,0.05,300,6,randomSeed);	

	//Physics Particle Construction
	//particleNode = new PhysicsNode*[RAW_WIDTH];
	//for(int i = 0; i < RAW_WIDTH;i++){ particleNode[i] = new PhysicsNode[RAW_HEIGHT]; }
	
	//HeightMap Generation
	std::ifstream file (name.c_str(),ios::binary);
	if (! file ) { return ; }

	numVertices = RAW_WIDTH*RAW_HEIGHT;
	numIndices = (RAW_WIDTH-1)*(RAW_HEIGHT-1)*6;
	vertices = new Vector3[numVertices];
	textureCoords = new Vector2[numVertices];
	indices = new GLuint[numIndices];
	unsigned char * data = new unsigned char [numVertices];
	file.read((char*)data,numVertices*sizeof(unsigned char));
	file.close();

	for (int x = 0; x < RAW_WIDTH ; ++ x ) {
		for (int z = 0; z < RAW_HEIGHT ; ++ z ) {

		//Simple Height Map Reshape
			//float mid = RAW_WIDTH/2 - abs((RAW_WIDTH-x)-(RAW_WIDTH/2));
			//float midz = RAW_WIDTH/2 - abs((RAW_WIDTH-z)-(RAW_WIDTH/2));
			//mid = mid/(RAW_WIDTH/2);
			//midz = midz/(RAW_WIDTH/2);
			//mid = mid* midz;
			//mid += 0.2;
			//mid *= 3;

		//Buffer Data into array 
			int offset = ( x * RAW_WIDTH ) + z ;
			
			//if(!usePerlinNoise) {
				vertices[offset] = Vector3(x*HEIGHTMAP_X , data [ offset ] * HEIGHTMAP_Y /** mid*/ , z * HEIGHTMAP_Z );
			//} else {
			//	double heightNoise = perlinNoiseF1->GetHeight(x,z);
			//	vertices[offset] = Vector3(x*HEIGHTMAP_X ,heightNoise * HEIGHTMAP_Y * mid + 100, z * HEIGHTMAP_Z );
			//}
			textureCoords[offset] = Vector2(x*HEIGHTMAP_TEX_X , z * HEIGHTMAP_TEX_Z );
			
		////Setup ParticleNode
		//	particleNode[x][z].m_position = vertices[offset];			
		//	particleNode[x][z].s1->m_pos = vertices[offset];			
		//	particleNode[x][z].boundingBox->m_pos = vertices[offset];

		//	particleNode[x][z].SetInverseMass(HEIGHTMAP_MASS);

		//	particleNode[x][z].s1->m_radius = 10.0f;
		//	particleNode[x][z].boundingBox->halfdims = 10.0f;
		//	particleNode[x][z].boundingBox->useSubList = false;			
		//	particleNode[x][z].boundingBox->useOctTree = false;			
		//	particleNode[x][z].isSlept = true;
		//	//if(offset%HEIGHTMAP_COLLSION_SEPARATION == 0){particleList.push_back(&particleNode[x][z]);}

		////Max and Min XYZ Detection
		//	if(x == 0 && z == 0){
		//		maxXYZ = particleNode[x][z].m_position;
		//		minXYZ = particleNode[x][z].m_position;
		//	} else {
		//		if(maxXYZ.x < particleNode[x][z].m_position.x){ maxXYZ.x = particleNode[x][z].m_position.x; }
		//		if(maxXYZ.y < particleNode[x][z].m_position.y){	maxXYZ.y = particleNode[x][z].m_position.y; }
		//		if(maxXYZ.z < particleNode[x][z].m_position.z){	maxXYZ.z = particleNode[x][z].m_position.z; }
		//		if(minXYZ.x > particleNode[x][z].m_position.x){ minXYZ.x = particleNode[x][z].m_position.x; }
		//		if(minXYZ.y > particleNode[x][z].m_position.y){ minXYZ.y = particleNode[x][z].m_position.y; }
		//		if(minXYZ.z > particleNode[x][z].m_position.z){ minXYZ.z = particleNode[x][z].m_position.z; }
		//	}
		}
	}
	//delete data ;
	//delete perlinNoiseF1;

	//Indices Bonding Code
	numIndices = 0;
	for (int x = 0; x < RAW_WIDTH -1; ++ x ) {
		for (int z = 0; z < RAW_HEIGHT -1; ++ z ) {
			int a = (x * ( RAW_WIDTH )) + z ;
			int b = ((x +1) * ( RAW_WIDTH )) + z ;
			int c = ((x +1) * ( RAW_WIDTH )) + ( z +1);
			int d = (x * ( RAW_WIDTH )) + ( z +1);
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

	//Custom Normal Calculation for Collision Impulse
	for (int i = 0; i < RAW_WIDTH -1; ++ i ) {
		for (int j = 0; j < RAW_HEIGHT-1 ; ++ j ) {
			int origin = (i * ( RAW_WIDTH )) + j ; //Origin
			int top = ((i +1) * ( RAW_WIDTH )) + j ; //Top				
			int bottom = ((i -1) * ( RAW_WIDTH )) + j ; //Bottom
			int left = ( i * ( RAW_WIDTH )) + ( j +1); //Left
			int right = ( i * ( RAW_WIDTH )) + ( j -1); //Right
			Vector3 surfaceNormal = normal[origin];
			
			if(top >= 0 && top < numVertices){ surfaceNormal += normal[top];}
			if(bottom >= 0 && bottom < numVertices){ surfaceNormal += normal[bottom];}
			if(left >= 0 && left < numVertices){ surfaceNormal += normal[left]; }
			if(right >= 0 && right < numVertices){ surfaceNormal += normal[right];}
			surfaceNormal.Normalise();
			
			////Set custom facing normal for particle
			//particleNode[i][j].useSelfNormal = true;
			//particleNode[i][j].collisionNormal = surfaceNormal;			
		}
	}

	////HeightMap Done Generation,Structure particle node into Oct Tree
	//mapTree = new OctTree(Vector3(minXYZ+((maxXYZ-minXYZ)/2)),Vector3((maxXYZ-minXYZ)/2),1);
	//for(vector<PhysicsNode*>::iterator i = particleList.begin(); i != particleList.end(); ++i) { mapTree->add((*i));}
	//
	////Create Bounding Box For HeightMap and Set Collision Type
	//physicsBound = new PhysicsNode();	
	//physicsBound-> boundingBox->m_pos = Vector3(minXYZ+((maxXYZ-minXYZ)/2));
	//physicsBound-> boundingBox->halfdims = Vector3((maxXYZ-minXYZ)/2);
	//physicsBound-> boundingBox->useSubList = false;
	//physicsBound-> boundingBox->useOctTree = true;
	//physicsBound-> boundingBox->ListPtr = mapTree;
	//physicsBound-> isSlept = true;

	////Add into Physics System
	//if(HEIGHTMAP_COLLIABLE)PhysicsSystem::GetPhysicsSystem().AddNode(physicsBound);
}

void HeightMap::Draw() {	

	//Assign Additional Texture for Multiple Layers.
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D,textureLayer[0]); 
	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D,textureLayer[1]);
	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D,textureLayer[2]);
	glActiveTexture(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_2D,textureLayer[3]);
	glActiveTexture(GL_TEXTURE6);
	glBindTexture(GL_TEXTURE_2D,textureLayer[4]);
	Mesh::Draw();
}