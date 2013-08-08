#pragma once
#include "../../nclgl/Mesh.h"


#define MILLION 1000000
#define INIT_NUMBER_OF_PARTICLE 0.5*MILLION

extern "C" {
	struct cudaParticle;
	//Constructor
	void cudaPS_initPS(int size,GLuint in_vbo,GLuint in_cbo);
	void cudaPS_destoryPS();
	//Cuda Draw and Update.
	int	 cudaPS_update(float msec,Vector3 pos);
	void cudaPS_bufferData(Vector3* vertices,Vector4* colours);
}

class ParticleSystem : public Mesh{
public:

	static void Initialise() { instance = new ParticleSystem(); }
	static void Destroy() { delete instance;  }
	static ParticleSystem& GetPhysicsSystem() { return *instance; }

	void init(int size){	
		resizeBuffer(size);
		texture = SOIL_load_OGL_texture(TEXTUREDIR"particle.tga",
			SOIL_LOAD_AUTO,SOIL_CREATE_NEW_ID,SOIL_FLAG_COMPRESS_TO_DXT);	
	}
	ParticleSystem(){
		init(INIT_NUMBER_OF_PARTICLE);
		cudaPS_initPS(INIT_NUMBER_OF_PARTICLE,bufferObject[VERTEX_BUFFER],bufferObject[COLOUR_BUFFER]);	
		bufferSize = INIT_NUMBER_OF_PARTICLE;				
	}
	ParticleSystem(int size){		
		init(size);
		cudaPS_initPS(size,bufferObject[VERTEX_BUFFER],bufferObject[COLOUR_BUFFER]);
		bufferSize = size;		
	}
	~ParticleSystem(void){
		cudaPS_destoryPS();
	}

	//Mesh Function
	void Update(float msec,Vector3 pos){
		cudaPS_update(msec,pos);
	}
	virtual void Draw();

protected:
	// Particle System
	static ParticleSystem* instance;

	void resizeBuffer(int size);

	int bufferSize;
};
