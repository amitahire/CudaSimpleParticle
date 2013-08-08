#pragma comment(lib, "nclgl.lib") 
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")

//Setting Cuda path.
#define CUDA_PATH "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v5.0\\"

#pragma comment(lib, CUDA_PATH "lib\\Win32\\cuda.lib")
#pragma comment(lib, CUDA_PATH "lib\\Win32\\cudart.lib")


#include "../../nclgl/Window.h"
//#include "KeyboardMouseManager.h"
#include "MyGame.h"

/*
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include "PhysicsNode.h"
#include "CollisionPairs.h"

PhysicsNode* cuda_node = NULL;
PhysicsNode* cuda_result = NULL;*/

int Quit(bool pause = false, const string &reason = "") {	
	//Deconstruct Static Systems
		
	PhysicsSystem::Destroy();
	Window::Destroy();
	Renderer::Destroy();	
	//KeyboardMouseManager::Destroy();

	//cudaFree(cuda_node);
	//cudaDeviceReset();

	if(pause) {
		std::cout << reason << std::endl;
		system("PAUSE"); 
	}
	return 0;
}
//
/*
__device__ void editCUDA(int* out,int index, int value){
	out[index] = value;
	//invec.push_back(value);
	
}

__global__ void hiCUDA(int* out){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	editCUDA(out,x*10+y,y*100 + x);
}

extern "C" void hiCUDAC(int* host_out){   
    const int N = 10;
	const int N2 = N*N;
	int* device_out = NULL;
	cudaMalloc((void **)&device_out,N2*sizeof(int));
	cudaMemset(device_out,0,N2*sizeof(int));
	cout << N2*sizeof(int) << endl;
	cudaMemcpy(device_out,host_out,N2*sizeof(int),cudaMemcpyHostToDevice);

	//thrust::device_vector<int> dev_vec;

	dim3 arraytwo(N,N);
	
	hiCUDA<<<1, arraytwo>>>(device_out);	

	cudaMemcpy(host_out,device_out,N2*sizeof(int),cudaMemcpyDeviceToHost);

	for(int i = 0; i < N2; i++){
		cout << host_out[i] << endl;	
	}

	//cout << "HEY:" << dev_vec.size() <<endl;

	cudaFree(device_out);
}

__global__ void heyCuda_BoardPhaseCollision(PhysicsNode* nodeList,PhysicsNode* resultList,int size,thrust::device_vector<int> inint){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	
}

extern "C" void hiCuda_BoardPhaseCollision(vector<PhysicsNode*>* nodeList,vector<PhysicsNode*>* resultList){
	//Setup Data Use
	resultList->clear();
	int size = nodeList->size();
	cudaMemcpy(cuda_node,(*nodeList)[0],size*sizeof(PhysicsNode),cudaMemcpyHostToDevice);
	cudaMemset(cuda_result,0,10000*sizeof(PhysicsNode));
	int block = (size/512)+1;
 	thrust::device_vector<int> temp;
	heyCuda_BoardPhaseCollision<<<block,512>>>(cuda_node,cuda_result,size,temp);
	
}
*/
	
int main() {
	//int N = 10000;
	//cudaMalloc((void**)cuda_node,N*sizeof(PhysicsNode));
	//cudaMalloc((void**)cuda_result,N*sizeof(float));

	/*const int N = 10;
	const int N2 = N*N;
	int iaa[N2];
	for(int i = 0; i < N2 ; i++){
		iaa[i] = 0;
	}
	hiCUDAC(iaa);
	*/
	
	PhysicsSystem::Initialise();

	if(!Window::Initialise("Game Technologies", 1280,800,false)) {return Quit(true, "Window failed to initialise!");}
			
	if(!Renderer::Initialise()) { return Quit(true, "Renderer failed to initialise!"); }
	
	
	MyGame* game = new MyGame();
	
	Window::GetWindow().LockMouseToWindow(true);
	Window::GetWindow().ShowOSPointer(false);

	while(Window::GetWindow().UpdateWindow() && !Window::GetKeyboard()->KeyDown(KEYBOARD_ESCAPE)){
		float msec = Window::GetWindow().GetTimer()->GetTimedMS();	
		game->UpdateCore(msec);
		game->UpdateGame(msec);
		
	}
	delete game;

	return Quit();
	
	
}