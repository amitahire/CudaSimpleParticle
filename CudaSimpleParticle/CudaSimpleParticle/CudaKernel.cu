#ifndef CudaKernel_h
#define CudaKernel_h

/*Standard Library Include*/
#include <iostream>
#include <vector>
#include <stdio.h>
using namespace std;

/*Thrust Library*/
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"
#include "thrust\for_each.h"
#include "thrust\reduce.h"

/*Cuda Library Include*/
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

/*Physics System*/
#include "PhysicsNode.h"
//#include "CollisionInfo.h"

/*Step1: Implement a basic Particle System On CUDA*/
struct Particle {
	Vector3 position;
	Vector4 colour;
	Vector3 direction;
};

extern "C" {

class ParticleSystem : public Mesh{
public:
	ParticleSystem(void);
	~ParticleSystem(void);

	void Update(float msec);

	virtual void Draw();

	/*How often we spit out some new particles!*/
	float	GetParticleRate()				{return particleRate;}
	void	SetParticleRate(float rate)		{particleRate = rate;}

	/*How long each particle lives for!*/
	float	GetParticleLifetime()			{return particleLifetime;}
	void	SetParticleLifetime(float life) {particleLifetime = life;}

	/*How big each particle will be!*/
	float	GetParticleSize()				{return particleSize;}
	void	SetParticleSize(float size)		{particleSize = size;}

	/*How much variance of the direction axis each particle can have when being launched. 
	  Variance of 0 = each particle's direction is = to the emitter direction. 
	  Variance of 1 = Each particle can go inany direction (with a slight bias towards the emitter direction)*/
	float	GetParticleVariance()				{return particleVariance;}
	void	SetParticleVariance(float variance) {particleVariance = variance;}

	/*Linear velocity of the particle*/
	float	GetParticleSpeed()				{return particleSpeed;}
	void	SetParticleSpeed(float speed)	{particleSpeed = speed;}

	/*How many particles does the emitter launch when it hits it's update time*/
	int		GetLaunchParticles()			{return numLaunchParticles;}
	void	SetLaunchParticles(int num)		{numLaunchParticles = num;}

	/*Launch direction of the particles*/
	void	SetDirection(const Vector3 dir) {initialDirection = dir;}
	Vector3 GetDirection()					{return initialDirection;}

private:

	Particle* GetFreeParticle();

	void	ResizeArrays();

	float particleRate;
	float particleLifetime;
	float particleSize;
	float particleVariance;
	float particleSpeed;
	int	  numLaunchParticles;

	Vector3 initialDirection;

	float nextParticleTime;

	int largestSize;

	thrust::device_vector<Particle*> d_particleList;
	thrust::device_vector<Particle*> d_freeList;

	thrust::host_vector<Particle*> particles;
	thrust::host_vector<Particle*> freeList;
};
	
}

#endif