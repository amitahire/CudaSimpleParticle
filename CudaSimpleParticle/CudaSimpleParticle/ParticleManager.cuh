#ifndef PARTICLEMANAGER_H
#define PARTICLEMANAGER_H

#ifndef __CUDA_RUNTIME_H__
#define __CUDA_RUNTIME_H__
#endif

#include <iostream>
#include <vector>
#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <math.h>
using namespace std;

#include "../../nclgl/Vector3.h"
#include "../../nclgl/Vector4.h"


//Thrust Library
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"
#include "thrust\device_ptr.h"
#include "thrust\sort.h"

//Cuda Library Include
#include "cuda_runtime.h"
#include "cuda.h"
#include "curand.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

/*Fix GL.h include Before GLEW.H*/ /*For Cuda_GL_Interop.H*/
#include <GL/glew.h>
#include <GL/wglew.h>
/*Cuda OpenGL Map Buffer Include*/
#include "cuda_gl_interop.h"


/*Define Configuration*/
#define CUDA_DEBUG_PRINTF true
#define NUM_OF_THREAD 256
#define RAND() ((rand()%101)/100.0f)

extern "C" {

	    void cudaInit(int argc, char **argv);
		void cudaGLInit(int argc, char **argv);

		//void initParticle(Vector4* in_cbo, Vector3* in_vbo, int size);

		//Host device functions.
		void init_Variable();
		void cudaPS_releaseBuffer(struct cudaGraphicsResource ** cudaResource);
		void cudaPS_bindBuffer(void** cudaPointer, struct cudaGraphicsResource ** cudaResource);
		void cudaPS_bindAllBuffers();
		void cudaPS_releaseAllBuffers();

		void cudaPS_initBuffer();
		void cudaPS_setupBuffer(GLuint in_buffer, struct cudaGraphicsResource ** cudaResource);
		void cudaPS_setupAllBuffers(GLuint in_vbo,GLuint in_cbo);
		void cudaPS_unregisterAllBuffers();
		void calLaunchThread(int &num_block,int &num_thread, int num_objects);

		//Constructor & Deconstructor.
		void cudaPS_initPS(int size,GLuint in_vbo,GLuint in_cbo);
		void cudaPS_destoryPS();

		//Kernal Function.
		void cudaPS_update(float msec,Vector3 pos);

		//Random number Gennerators. -Not working.
		curandStatus_t CURANDAPI 
		curandCreateGeneratorHost(curandGenerator_t *generator, curandRngType_t rng_type);


	// Particle Engine Setup
	struct cudaParticleSetup {

		float cuda_pLife;
		float cuda_pSize;
		float cuda_pVariance;
		float cuda_pSpeed;
		float cuda_pRate;			//Particle System Spawn Time
		int	  cuda_pLaunchNumber;	//Individual ParticleSystem Time Count*/
		float cuda_pNextTime;
	};

	// Positioning particles.
	struct cudaParticleModule  {
		cudaParticleSetup *h_setup;
		Vector3 *posRef;
	};
	
	//Particle Struct
	struct cudaParticle {
		float3 position;
		float3 direction; 
		float4 colour;
		__host__ __device__ cudaParticle(){
			position = make_float3(0,0,0);
			direction = make_float3(0,0,0);
			colour = make_float4(0,0,0,0);

		}
		__host__ __device__ cudaParticle(float3 in_pos,float3 in_dir,float4 in_colour):
			position(in_pos),
			direction(in_dir),
			colour(in_colour){}
	};

	
}

//Vertex and Colour buffers.
GLuint openGL_Vbo;
struct cudaGraphicsResource* cudaPs_VboResource;
Vector3* cudaPs_VboPtr;


GLuint openGL_Cbo;
struct cudaGraphicsResource* cudaPs_CboResource;
Vector4* cudaPs_CboPtr;

/*Device Variable*/
thrust::device_vector<cudaParticle> d_particle;
thrust::host_vector<cudaParticle> h_particle;

//Pointer to Vector array (For Kernel)
cudaParticle* particlePtr;
curandState* randomStates;

//Size Configuration
int d_numParticle;
int d_maxParticle;

//Thread Configuration for CUDA.
int num_block;
int num_thread;

//Particle System Configuration
float particleRate;
float particleLifetime;
float particleSize;
float particleVariance;
float particleSpeed;
int	  numLaunchParticles;

float nextParticleTime;

#endif
