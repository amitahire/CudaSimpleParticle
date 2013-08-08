#ifndef CudaKernel_h
#define CudaKernel_h

#include "ParticleManager.cuh"
#include <cuda.h>
#include <curand.h>

#define NUM_OF_THREAD 256

#define NUM_OF_RAN_BLOCK 512

#define NUM_OF_RAN_SET 2

extern "C"{
	
	//Helper function provided by NVIDIA to init cuda
    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }
}

////GLOBAL////////////////////////////////////////////////////////

//Cuda Kernel Function
__global__ void initParticle(Vector4* in_cbo, Vector3* in_vbo, int size){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < size){
		in_vbo[tid].x = (tid%4096)*500;
		in_vbo[tid].y = 0.0f;
		in_vbo[tid].z = ((int)(tid/4096))*15*500;

		in_cbo[tid].x = 1.0f;
		in_cbo[tid].y = 0.0f;
		in_cbo[tid].z = 0.0f;
		in_cbo[tid].w = 0.0f;
	}
}


//HOST//////////////////////////////////////////

__host__ void calLaunchThread(int &num_block,int &num_thread, int num_objects){
	num_block = (num_objects/NUM_OF_THREAD)+1;
	num_thread = (num_objects < NUM_OF_THREAD)?num_objects:NUM_OF_THREAD;
}

__host__ void init_Variable(){

	particleRate		= 90.0f;
	particleLifetime	= 20000; 
	particleSize		= 35.0f;
	particleVariance	= 20.0f;
	particleSpeed		= 0.1f;

	nextParticleTime	= 0.0f;
	numLaunchParticles	= 400;
	d_numParticle		= 0;

	/*
	//Random Gen.
	size_t	n = 100;
	curandGenerator_t	gen;
	float *devData, *hostData;

	//Allocate n floats on host
	hostData = (float *)calloc(n, sizeof(float));

	// Allocate n floats on device.
    CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(float)));

    // Create pseudo-random number generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

	// Generate n floats on device 
    CURAND_CALL(curandGenerateUniform(gen, devData, n));

    // Copy device memory to host
    CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(float),
        cudaMemcpyDeviceToHost));

	// Show result
    for(size_t i = 0; i < n; i++) {
        printf("%1.4f ", hostData[i]);
    }
    printf("\n");

		// Cleanup.
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    free(hostData);    
    //return EXIT_SUCCESS;

	*/
}


//Buffer Binding.

__host__ void cudaPS_releaseBuffer(struct cudaGraphicsResource ** cudaResource){
	cudaGraphicsUnmapResources(1, cudaResource, 0); // give access authority of vbo1 back to openGL  
}

__host__ void cudaPS_bindBuffer(void** cudaPointer, struct cudaGraphicsResource ** cudaResource){
	size_t num_bytes;
	cudaGraphicsMapResources(1, cudaResource, 0);	
}

__host__ void cudaPS_bindAllBuffers(){
	cudaPS_bindBuffer((void **)&cudaPs_VboPtr,&cudaPs_VboResource);
	cudaPS_bindBuffer((void **)&cudaPs_CboPtr,&cudaPs_CboResource);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&cudaPs_VboPtr, &num_bytes, cudaPs_VboResource);
	cudaGraphicsResourceGetMappedPointer((void **)&cudaPs_CboPtr, &num_bytes, cudaPs_CboResource);
}

__host__ void cudaPS_releaseAllBuffers(){
	cudaPS_releaseBuffer(&cudaPs_VboResource);
	cudaPS_releaseBuffer(&cudaPs_CboResource);
}

//Setting up buffer & destroying.
__host__ void cudaPS_setupBuffer(GLuint in_buffer, struct cudaGraphicsResource ** cudaResource){
	cudaGraphicsGLRegisterBuffer(cudaResource,in_buffer,cudaGraphicsMapFlagsNone);
}

__host__ void cudaPS_setupAllBuffers(GLuint in_vbo,GLuint in_cbo){

	openGL_Vbo = in_vbo;
	openGL_Cbo = in_cbo;
	cudaPS_setupBuffer(in_vbo,&cudaPs_VboResource);
	cudaPS_setupBuffer(in_cbo,&cudaPs_CboResource);
}

__host__ void cudaPS_unregisterAllBuffers(){
	cudaGraphicsUnregisterResource(cudaPs_VboResource);
	cudaGraphicsUnregisterResource(cudaPs_CboResource);
}

__host__ void cudaPS_initBuffer(){
	cudaPS_bindAllBuffers();
	calLaunchThread(num_block,num_thread,d_maxParticle);
	initParticle<<<num_block,num_thread>>>(cudaPs_CboPtr,cudaPs_VboPtr,d_maxParticle);		
	cudaThreadSynchronize();
	cudaPS_releaseAllBuffers();
}


__global__ void particleUpdate(Vector4* in_cbo, Vector3* in_vbo, cudaParticle* in_particle, int size, float msec, float particleSpeed,float particleLifetime, Vector3 pos){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//Random Gen.
	//curandState		gen;
	//retrieveRandomState(gen, globalState);


	if(tid < size){
			in_cbo[tid].x = 1.0f;
			in_cbo[tid].y = 0.0f;
			in_cbo[tid].z = 0.0f;
			in_cbo[tid].w = 1.0f;

			float3 direction = make_float3(pos.x-in_vbo[tid].x,pos.y-in_vbo[tid].y,pos.z-in_vbo[tid].z);
		
			float length = cbrt((direction.x*direction.x)+(direction.y*direction.y)+(direction.z*direction.z));
			//float length = sqrt((direction.x*direction.x)+(direction.y*direction.y)+(direction.z*direction.z));

						
			// Star shaped shape.
			if(length >= 50.0f)	{
				length = (1.0f / length);
				direction.x = direction.x * length;
				direction.y = direction.y * length;
				direction.z = direction.z * length;
			}

			if (direction.x <= 100.0f || direction.y <= 100.0f || direction.z <= 100.0f) {
				length += 100.0f;
			}

			in_vbo[tid].x = in_vbo[tid].x + direction.x*10;
			in_vbo[tid].y = in_vbo[tid].y + direction.y*10;
			in_vbo[tid].z = in_vbo[tid].z + direction.z*10;			
	}
}



//External C Function.
extern "C" {

	void cudaPS_initPS(int size,GLuint in_vbo,GLuint in_cbo){
		//Init, reserve certain amount of particle attribute
		d_particle.reserve(size);
		h_particle.reserve(size);
		d_maxParticle = size;

		//Setup Buffer Object
		cudaPS_setupAllBuffers(in_vbo,in_cbo);
		cudaPS_initBuffer();
		init_Variable();
	}
	void cudaPS_destoryPS(){
		cudaPS_unregisterAllBuffers();
		//cudaFree(randomStates);
	}
		
	//Kernel Function
	void cudaPS_update(float msec,Vector3 pos){
		//Bind Buffer to Write
		cudaPS_bindAllBuffers();

		//Core Particle Update
		calLaunchThread(num_block,num_thread,d_maxParticle);
		particleUpdate<<<num_block,num_thread>>>(cudaPs_CboPtr,cudaPs_VboPtr,particlePtr,d_maxParticle,msec,particleSpeed,6000,pos);		
		cudaThreadSynchronize();
		
		cudaPS_releaseAllBuffers();	

	}	
};

#endif