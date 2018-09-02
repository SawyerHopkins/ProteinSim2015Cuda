# ProteinSim2015CUDA
A langevin dynamics engine for the GPU.

## Info

PSim2015 is a particle dynamics engine being developed in the Kansas State University Physics Department. 

This project is an experimental fork of [ProteinSim2015](../../../ProteinSim2015). To see usage and implementation refer to the original ProteinSim2015. Note this project lags behind the upstream and may not have all features implemented.

## Custom Forces

If you need a force or system of forces that do not exist within the project, you can create and import your own. This project manages the CUDA kernal's for particle dynamics simulations internally making it possible to run custom built forces and code on a CUDA offload device with almost no CUDA knowledge at all.

### 1. Create a new folder for the force.

In this folder create the necessary C++ files and either copy the PSim header files to this folder, or link the `ClusteredCore/include` folder with g++/make.

### 2. Create the force class.

Create a new class which inherits/implements the `PSim::IForce` interface.
```cpp
#include "forceManager.h"
class myForce : public PSim::IForce
```
### 3. Create the force factory.

Since this is c++, we need to make a c compatible factory to load this as an external library. The factories name must be named as shown.

```cpp
__global__
void buildKernel(physics::IForce** force, float* vars)
{
	(*force) = new myForce(vars);
}

//Class factories.
extern "C" physics::IForce* getForce(configReader::config* cfg)
{
	return new myForce(cfg);
}
```
You will also need to include these factories as well to interact with the force.
```cpp
__global__
void testKernel(physics::IForce** force)
{
	(*force)->cudaTest();
}

__global__
void accelerationKernel(physics::IForce** force, int* nPart, int* boxSize, int* cellScale ,float* time, simulation::cell* cells, simulation::particle* items)
{
	(*force)->getAcceleration(nPart, boxSize, cellScale, time, cells, items);
}

//Class factories.
extern "C" void getCudaForce(physics::IForce** force, float* vars)
{
	buildKernel<<<1,1>>>(force,vars);
	cudaDeviceSynchronize();
	std::string err = cudaGetErrorString(cudaGetLastError());
}

//Class factories.
extern "C" void runCudaTest(physics::IForce** force)
{
	testKernel<<<1,1>>>(force);
	cudaDeviceSynchronize();
	std::string err = cudaGetErrorString(cudaGetLastError());
}

//Class factories.
extern "C" void runAcceleration(physics::IForce** force, int* nPart, int* boxSize, int* cellScale ,float* time, simulation::cell* cells, simulation::particle* items, int numThreads)
{
	accelerationKernel<<<numThreads,1>>>(force, nPart, boxSize, cellScale, time, cells, items);
	cudaDeviceSynchronize();
	std::string err = cudaGetErrorString(cudaGetLastError());
}
```

### 4. Implement the interface contract

The `PSim::IForce` interface required on function `nextSystem` to be implemented. This function will have acess to the current time, the particle system, and the cell manager. To see how use the cell and particle systems checkout the documentation on the project website. Optionally take a look at LJPotential.cu.

The project manages the CUDA kernels automatically, there is no special work to do be done. Just prefix all functions in the class with the `__device__` compiler directive to allow it to be run on the offload device.

### 5. Usage Notes

The input parameter `numThreads` in `runAcceleration` factory is not the `threads` option in the config file. 
