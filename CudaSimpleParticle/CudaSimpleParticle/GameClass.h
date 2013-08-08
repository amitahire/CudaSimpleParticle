/******************************************************************************
Class:GameClass

Author:Rich Davison	<richard.davison4@newcastle.ac.uk> and Amit Ahire<amitahire@gmail.com>
Description: A very simple 'base class' with which to base your game projects
on. This will simply update all of the 'sub systems' of the program


-_-_-_-_-_-_-_,------,   
_-_-_-_-_-_-_-|   /\_/\   NYANYANYAN
-_-_-_-_-_-_-~|__( ^ .^) /
_-_-_-_-_-_-_-""  ""   

*//////////////////////////////////////////////////////////////////////////////

#pragma once
#include "Renderer.h"
#include "PhysicsSystem.h"
#include "GameEntity.h"

#include <vector>

#define RENDER_HZ	60
#define PHYSICS_HZ	120

#define PHYSICS_TIMESTEP (1000.0f / (float)PHYSICS_HZ)

class GameClass	{
public:
	GameClass();
	~GameClass(void);

	virtual void UpdateCore(float msec);

	virtual void UpdateGame(float msec) = 0;

	static GameClass& GetGameClass() { return *instance;}

protected:
	float renderCounter;
	float physicsCounter;

	vector<GameEntity*> allEntities;
	//vector<GameEntity*> ProjList;

  	void ClearNodeList() { allEntities.clear();}

	Camera* gameCamera;
	static GameClass* instance;
};

