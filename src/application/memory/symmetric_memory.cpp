// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "mori/application/memory/symmetric_memory.hpp"

#include "hip/hip_runtime.h"
#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/core/core.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace application {

SymmMemManager::SymmMemManager(BootstrapNetwork& bootNet, Context& context)
    : bootNet(bootNet), context(context) {}

SymmMemManager::~SymmMemManager() {
  while (!memObjPool.empty()) {
    DeregisterSymmMemObj(memObjPool.begin()->first);
  }
}

SymmMemObjPtr SymmMemManager::HostMalloc(size_t size, size_t alignment) {
  void* ptr = nullptr;
  int status = posix_memalign(&ptr, alignment, size);
  assert(!status);
  memset(ptr, 0, size);
  return RegisterSymmMemObj(ptr, size);
}

void SymmMemManager::HostFree(void* localPtr) {
  free(localPtr);
  DeregisterSymmMemObj(localPtr);
}

SymmMemObjPtr SymmMemManager::Malloc(size_t size) {
  void* ptr = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&ptr, size));
  HIP_RUNTIME_CHECK(hipMemset(ptr, 0, size));
  return RegisterSymmMemObj(ptr, size);
}

SymmMemObjPtr SymmMemManager::ExtMallocWithFlags(size_t size, unsigned int flags) {
  void* ptr = nullptr;
  HIP_RUNTIME_CHECK(hipExtMallocWithFlags(&ptr, size, flags));
  HIP_RUNTIME_CHECK(hipMemset(ptr, 0, size));
  return RegisterSymmMemObj(ptr, size);
}

void SymmMemManager::Free(void* localPtr) {
  HIP_RUNTIME_CHECK(hipFree(localPtr));
  DeregisterSymmMemObj(localPtr);
}

SymmMemObjPtr SymmMemManager::RegisterSymmMemObj(void* localPtr, size_t size, bool heap_begin) {
  int worldSize = bootNet.GetWorldSize();
  int rank = bootNet.GetLocalRank();

  SymmMemObj* cpuMemObj = new SymmMemObj();
  cpuMemObj->localPtr = localPtr;
  cpuMemObj->size = size;

  // Exchange pointers
  cpuMemObj->peerPtrs = static_cast<uintptr_t*>(calloc(worldSize, sizeof(uintptr_t)));
  bootNet.Allgather(&localPtr, cpuMemObj->peerPtrs, sizeof(uintptr_t));
  // cpuMemObj->peerPtrs[rank] = reinterpret_cast<uintptr_t>(cpuMemObj->localPtr);

  // P2P context: exchange ipc mem handles
  hipIpcMemHandle_t handle;
  HIP_RUNTIME_CHECK(hipIpcGetMemHandle(&handle, localPtr));
  cpuMemObj->ipcMemHandles =
      static_cast<hipIpcMemHandle_t*>(calloc(worldSize, sizeof(hipIpcMemHandle_t)));
  bootNet.Allgather(&handle, cpuMemObj->ipcMemHandles, sizeof(hipIpcMemHandle_t));
  for (int i = 0; i < worldSize; i++) {
    if (context.GetTransportType(i) != TransportType::P2P) continue;
    if (i == rank) continue;

    HIP_RUNTIME_CHECK(hipIpcOpenMemHandle(reinterpret_cast<void**>(&cpuMemObj->peerPtrs[i]),
                                          cpuMemObj->ipcMemHandles[i],
                                          hipIpcMemLazyEnablePeerAccess));
  }

  // Rdma context: set lkey and exchange rkeys
  cpuMemObj->peerRkeys = static_cast<uint32_t*>(calloc(worldSize, sizeof(uint32_t)));
  cpuMemObj->peerRkeys[rank] = 0;
  RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
  if (rdmaDeviceContext) {
    application::RdmaMemoryRegion mr = rdmaDeviceContext->RegisterRdmaMemoryRegion(localPtr, size);
    cpuMemObj->lkey = mr.lkey;
    cpuMemObj->peerRkeys[rank] = mr.rkey;
  }
  bootNet.Allgather(&cpuMemObj->peerRkeys[rank], cpuMemObj->peerRkeys, sizeof(uint32_t));

  // Copy memory object to GPU memory, we need to access it from GPU directly
  SymmMemObj* gpuMemObj;
  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj, sizeof(SymmMemObj)));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj, cpuMemObj, sizeof(SymmMemObj), hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerPtrs, sizeof(uintptr_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerPtrs, cpuMemObj->peerPtrs,
                              sizeof(uintptr_t) * worldSize, hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerRkeys, sizeof(uint32_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerRkeys, cpuMemObj->peerRkeys,
                              sizeof(uint32_t) * worldSize, hipMemcpyHostToDevice));

  SymmMemObjPtr result{cpuMemObj, gpuMemObj};
  if (!heap_begin) {
    memObjPool.insert({localPtr, result});
    return memObjPool.at(localPtr);
  } else {
    return result;
  }
}

void SymmMemManager::DeregisterSymmMemObj(void* localPtr) {
  if (memObjPool.find(localPtr) == memObjPool.end()) return;

  RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
  if (rdmaDeviceContext) rdmaDeviceContext->DeregisterRdmaMemoryRegion(localPtr);

  SymmMemObjPtr memObjPtr = memObjPool.at(localPtr);
  free(memObjPtr.cpu->peerPtrs);
  free(memObjPtr.cpu->peerRkeys);
  free(memObjPtr.cpu->ipcMemHandles);
  free(memObjPtr.cpu);
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu->peerPtrs));
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu->peerRkeys));
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu));

  memObjPool.erase(localPtr);
}

SymmMemObjPtr SymmMemManager::HeapRegisterSymmMemObj(void* localPtr, size_t size,
                                                     SymmMemObjPtr* heapObj) {
  int worldSize = bootNet.GetWorldSize();
  int rank = bootNet.GetLocalRank();

  SymmMemObj* cpuMemObj = new SymmMemObj();
  cpuMemObj->localPtr = localPtr;
  cpuMemObj->size = size;

  // Calculate offset from heap base
  uintptr_t heapBase = reinterpret_cast<uintptr_t>(heapObj->cpu->localPtr);
  uintptr_t localAddr = reinterpret_cast<uintptr_t>(localPtr);
  size_t offset = localAddr - heapBase;

  cpuMemObj->peerPtrs = static_cast<uintptr_t*>(calloc(worldSize, sizeof(uintptr_t)));
  for (int i = 0; i < worldSize; i++) {
    cpuMemObj->peerPtrs[i] = heapObj->cpu->peerPtrs[i] + offset;
  }

  cpuMemObj->ipcMemHandles =
      static_cast<hipIpcMemHandle_t*>(calloc(worldSize, sizeof(hipIpcMemHandle_t)));
  memcpy(cpuMemObj->ipcMemHandles, heapObj->cpu->ipcMemHandles,
         sizeof(hipIpcMemHandle_t) * worldSize);

  cpuMemObj->peerRkeys = static_cast<uint32_t*>(calloc(worldSize, sizeof(uint32_t)));
  memcpy(cpuMemObj->peerRkeys, heapObj->cpu->peerRkeys, sizeof(uint32_t) * worldSize);
  cpuMemObj->lkey = heapObj->cpu->lkey;

  SymmMemObj* gpuMemObj;
  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj, sizeof(SymmMemObj)));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj, cpuMemObj, sizeof(SymmMemObj), hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerPtrs, sizeof(uintptr_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerPtrs, cpuMemObj->peerPtrs,
                              sizeof(uintptr_t) * worldSize, hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerRkeys, sizeof(uint32_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerRkeys, cpuMemObj->peerRkeys,
                              sizeof(uint32_t) * worldSize, hipMemcpyHostToDevice));

  memObjPool.insert({localPtr, SymmMemObjPtr{cpuMemObj, gpuMemObj}});
  return memObjPool.at(localPtr);
}

void SymmMemManager::HeapDeregisterSymmMemObj(void* localPtr) {
  if (memObjPool.find(localPtr) == memObjPool.end()) return;

  // No need to deregister RDMA memory region - this is a sub-region of the heap

  SymmMemObjPtr memObjPtr = memObjPool.at(localPtr);
  free(memObjPtr.cpu->peerPtrs);
  free(memObjPtr.cpu->peerRkeys);
  free(memObjPtr.cpu->ipcMemHandles);
  free(memObjPtr.cpu);
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu->peerPtrs));
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu->peerRkeys));
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu));

  memObjPool.erase(localPtr);
}

SymmMemObjPtr SymmMemManager::Get(void* localPtr) const {
  if (memObjPool.find(localPtr) == memObjPool.end()) return SymmMemObjPtr{};
  return memObjPool.at(localPtr);
}

// VMM-based symmetric memory management implementation
bool SymmMemManager::IsVMMSupported() const {
  int vmm = 0;
  int currentDev = 0;
  hipError_t result = hipGetDevice(&currentDev);
  if (result != hipSuccess) {
    return false;  // Cannot get current device
  }

  result =
      hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, currentDev);
  return (result == hipSuccess && vmm != 0);
}

bool SymmMemManager::InitializeVMMHeap(size_t virtualSize, size_t chunkSize) {
  std::lock_guard<std::mutex> lock(vmmLock);

  if (vmmInitialized) {
    return true;  // Already initialized
  }

  if (!IsVMMSupported()) {
    return false;  // VMM not supported
  }

  // Determine optimal chunk size if not provided
  if (chunkSize == 0) {
    int currentDev = 0;
    hipError_t result = hipGetDevice(&currentDev);
    if (result != hipSuccess) {
      return false;
    }

    hipMemAllocationProp allocProp = {};
    allocProp.type = hipMemAllocationTypePinned;
    allocProp.location.type = hipMemLocationTypeDevice;
    allocProp.location.id = currentDev;

    size_t granularity = 0;
    result = hipMemGetAllocationGranularity(&granularity, &allocProp,
                                            hipMemAllocationGranularityRecommended);
    if (result == hipSuccess && granularity > 0) {
      chunkSize = granularity;
      size_t minGranularity = 0;
      result = hipMemGetAllocationGranularity(&minGranularity, &allocProp,
                                              hipMemAllocationGranularityMinimum);
      if (result == hipSuccess && minGranularity > 0) {
        vmmMinChunkSize = minGranularity;
      } else {
        vmmMinChunkSize = 4 * 1024;
      }
    } else {
      // Fallback to minimal granularity if recommended fails
      result = hipMemGetAllocationGranularity(&granularity, &allocProp,
                                              hipMemAllocationGranularityMinimum);
      if (result == hipSuccess && granularity > 0) {
        chunkSize = granularity;
        vmmMinChunkSize = granularity;
      } else {
        // Final fallback to 2MB
        chunkSize = 2 * 1024 * 1024;
        vmmMinChunkSize = 4 * 1024;
      }
    }
  }

  vmmVirtualSize = virtualSize;
  vmmChunkSize = chunkSize;
  vmmMaxChunks = virtualSize / chunkSize;
  MORI_APP_INFO(
      "VMM Heap Initialization: virtualSize = {}, chunkSize = {}, minChunkSize = {}, maxChunks = "
      "{}",
      vmmVirtualSize, vmmChunkSize, vmmMinChunkSize, vmmMaxChunks);
  // Reserve virtual address space
  hipError_t result = hipMemAddressReserve(&vmmVirtualBasePtr, virtualSize, 0, nullptr, 0);
  if (result != hipSuccess) {
    return false;
  }

  // Initialize chunk tracking
  vmmChunks.resize(vmmMaxChunks);

  // Initialize VMM heap unified RDMA registration
  int worldSize = bootNet.GetWorldSize();
  int rank = bootNet.GetLocalRank();

  uint32_t vmmHeapLkey{0};
  std::vector<uint32_t> vmmHeapRkeys;
  vmmHeapRkeys.resize(worldSize, 0);
  RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
  if (rdmaDeviceContext) {
    application::RdmaMemoryRegion mr =
        rdmaDeviceContext->RegisterRdmaMemoryRegion(vmmVirtualBasePtr, virtualSize);
    vmmHeapLkey = mr.lkey;
    vmmHeapRkeys[rank] = mr.rkey;
    vmmRdmaRegistered = true;
  }

  // Exchange RDMA keys among all PEs
  if (vmmRdmaRegistered) {
    bootNet.Allgather(&vmmHeapRkeys[rank], vmmHeapRkeys.data(), sizeof(uint32_t));
  }

  // Create SymmMemObjPtr for the entire VMM heap
  SymmMemObj* cpuHeapObj = new SymmMemObj();
  cpuHeapObj->localPtr = vmmVirtualBasePtr;
  cpuHeapObj->size = virtualSize;

  // Exchange virtual base pointers among all PEs
  cpuHeapObj->peerPtrs = static_cast<uintptr_t*>(calloc(worldSize, sizeof(uintptr_t)));
  bootNet.Allgather(&vmmVirtualBasePtr, cpuHeapObj->peerPtrs, sizeof(uintptr_t));

  // VMM doesn't need IPC handles - access is managed through hipMemSetAccess
  cpuHeapObj->ipcMemHandles =
      static_cast<hipIpcMemHandle_t*>(calloc(worldSize, sizeof(hipIpcMemHandle_t)));

  // Use unified RDMA keys
  cpuHeapObj->peerRkeys = static_cast<uint32_t*>(calloc(worldSize, sizeof(uint32_t)));
  if (vmmRdmaRegistered) {
    cpuHeapObj->lkey = vmmHeapLkey;
    memcpy(cpuHeapObj->peerRkeys, vmmHeapRkeys.data(), sizeof(uint32_t) * worldSize);
  } else {
    cpuHeapObj->lkey = 0;
    memset(cpuHeapObj->peerRkeys, 0, sizeof(uint32_t) * worldSize);
  }

  // Copy heap object to GPU memory
  SymmMemObj* gpuHeapObj;
  HIP_RUNTIME_CHECK(hipMalloc(&gpuHeapObj, sizeof(SymmMemObj)));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuHeapObj, cpuHeapObj, sizeof(SymmMemObj), hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuHeapObj->peerPtrs, sizeof(uintptr_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuHeapObj->peerPtrs, cpuHeapObj->peerPtrs,
                              sizeof(uintptr_t) * worldSize, hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuHeapObj->peerRkeys, sizeof(uint32_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuHeapObj->peerRkeys, cpuHeapObj->peerRkeys,
                              sizeof(uint32_t) * worldSize, hipMemcpyHostToDevice));

  // Store the VMM heap object
  vmmHeapObj = SymmMemObjPtr{cpuHeapObj, gpuHeapObj};

  vmmInitialized = true;
  return true;
}

void SymmMemManager::FinalizeVMMHeap() {
  std::lock_guard<std::mutex> lock(vmmLock);

  if (!vmmInitialized) {
    return;
  }

  // Deregister VMM heap's unified RDMA registration before cleanup
  if (vmmRdmaRegistered) {
    RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
    if (rdmaDeviceContext) {
      rdmaDeviceContext->DeregisterRdmaMemoryRegion(vmmVirtualBasePtr);
    }
    vmmRdmaRegistered = false;
  }

  // Free all allocated chunks
  for (size_t i = 0; i < vmmMaxChunks; ++i) {
    if (vmmChunks[i].isAllocated) {
      void* chunkPtr = static_cast<void*>(static_cast<char*>(vmmVirtualBasePtr) + i * vmmChunkSize);
      // Use the actual physical size that was mapped
      HIP_RUNTIME_CHECK(hipMemUnmap(chunkPtr, vmmChunks[i].physicalSize));
      HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[i].handle));
    }
  }

  // Free virtual address space
  if (vmmVirtualBasePtr) {
    HIP_RUNTIME_CHECK(hipMemAddressFree(vmmVirtualBasePtr, vmmVirtualSize));
    vmmVirtualBasePtr = nullptr;
  }

  // Clean up VMM heap object
  if (vmmHeapObj.IsValid()) {
    free(vmmHeapObj.cpu->peerPtrs);
    free(vmmHeapObj.cpu->peerRkeys);
    free(vmmHeapObj.cpu->ipcMemHandles);
    free(vmmHeapObj.cpu);
    HIP_RUNTIME_CHECK(hipFree(vmmHeapObj.gpu->peerPtrs));
    HIP_RUNTIME_CHECK(hipFree(vmmHeapObj.gpu->peerRkeys));
    HIP_RUNTIME_CHECK(hipFree(vmmHeapObj.gpu));
    vmmHeapObj = SymmMemObjPtr{nullptr, nullptr};
  }

  vmmChunks.clear();
  vmmMinChunkSize = 0;
  vmmInitialized = false;
}

SymmMemObjPtr SymmMemManager::VMMAllocChunk(size_t size, uint32_t allocType) {
  std::lock_guard<std::mutex> lock(vmmLock);

  if (!vmmInitialized) {
    MORI_APP_WARN("VMMAllocChunk failed: VMM heap not initialized");
    return SymmMemObjPtr{nullptr, nullptr};
  }

  size_t chunksNeeded = (size + vmmChunkSize - 1) / vmmChunkSize;

  // Find contiguous free chunks
  size_t startChunk = 0;
  bool found = false;
  for (size_t i = 0; i <= vmmMaxChunks - chunksNeeded; ++i) {
    bool canAllocate = true;
    for (size_t j = 0; j < chunksNeeded; ++j) {
      if (vmmChunks[i + j].isAllocated) {
        canAllocate = false;
        break;
      }
    }
    if (canAllocate) {
      startChunk = i;
      found = true;
      break;
    }
  }

  if (!found) {
    MORI_APP_WARN(
        "VMMAllocChunk failed: No contiguous free chunks available for size {} bytes (need {} "
        "chunks)",
        size, chunksNeeded);
    return SymmMemObjPtr{nullptr, nullptr};
  }

  // Allocate physical memory for each chunk
  int currentDev = 0;
  hipError_t result = hipGetDevice(&currentDev);
  if (result != hipSuccess) {
    MORI_APP_WARN("VMMAllocChunk failed: Cannot get current device, hipError: {}", result);
    return SymmMemObjPtr{nullptr, nullptr};
  }

  hipMemAllocationProp allocProp = {};
  allocProp.type = static_cast<hipMemAllocationType>(allocType);
  allocProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;  // Enable shareable handles
  allocProp.location.type = hipMemLocationTypeDevice;
  allocProp.location.id = currentDev;

  for (size_t i = 0; i < chunksNeeded; ++i) {
    size_t chunkIdx = startChunk + i;
    void* chunkPtr =
        static_cast<void*>(static_cast<char*>(vmmVirtualBasePtr) + chunkIdx * vmmChunkSize);

    // Calculate actual physical memory needed for this chunk
    size_t remainingSize = size - (i * vmmChunkSize);
    size_t physicalSize = std::max(std::min(remainingSize, vmmChunkSize), vmmMinChunkSize);

    // Create physical memory with actual needed size
    hipError_t result = hipMemCreate(&vmmChunks[chunkIdx].handle, physicalSize, &allocProp, 0);
    if (result != hipSuccess) {
      MORI_APP_WARN(
          "VMMAllocChunk failed: hipMemCreate failed for chunk {} with size {} bytes, allocType: "
          "{}, device: {}, hipError: {}",
          chunkIdx, physicalSize, allocType, currentDev, result);
      // Cleanup already allocated chunks
      for (size_t j = 0; j < i; ++j) {
        size_t cleanupIdx = startChunk + j;
        HIP_RUNTIME_CHECK(hipMemUnmap(
            static_cast<void*>(static_cast<char*>(vmmVirtualBasePtr) + cleanupIdx * vmmChunkSize),
            vmmChunks[cleanupIdx].physicalSize));
        HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[cleanupIdx].handle));
        vmmChunks[cleanupIdx].isAllocated = false;
        vmmChunks[cleanupIdx].physicalSize = 0;
      }
      return SymmMemObjPtr{nullptr, nullptr};
    }

    // Export shareable handle for cross-process sharing
    result = hipMemExportToShareableHandle((void*)&vmmChunks[chunkIdx].shareableHandle, 
                                           vmmChunks[chunkIdx].handle, 
                                           hipMemHandleTypePosixFileDescriptor, 0);
    if (result != hipSuccess) {
      MORI_APP_WARN(
          "VMMAllocChunk warning: hipMemExportToShareableHandle failed for chunk {}, hipError: {}. "
          "Cross-process sharing may not work.", 
          chunkIdx, result);
      vmmChunks[chunkIdx].shareableHandle = -1;
    }

    // Map physical memory to virtual address (map only the physical size)
    result = hipMemMap(chunkPtr, physicalSize, 0, vmmChunks[chunkIdx].handle, 0);
    if (result != hipSuccess) {
      MORI_APP_WARN(
          "VMMAllocChunk failed: hipMemMap failed for chunk {} at address {:p} with size {} bytes, "
          "hipError: {}",
          chunkIdx, chunkPtr, physicalSize, result);
      HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[chunkIdx].handle));
      // Cleanup already allocated chunks
      for (size_t j = 0; j < i; ++j) {
        size_t cleanupIdx = startChunk + j;
        HIP_RUNTIME_CHECK(hipMemUnmap(
            static_cast<void*>(static_cast<char*>(vmmVirtualBasePtr) + cleanupIdx * vmmChunkSize),
            vmmChunks[cleanupIdx].physicalSize));
        HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[cleanupIdx].handle));
        vmmChunks[cleanupIdx].isAllocated = false;
        vmmChunks[cleanupIdx].physicalSize = 0;
      }
      return SymmMemObjPtr{nullptr, nullptr};
    }

    vmmChunks[chunkIdx].isAllocated = true;
    vmmChunks[chunkIdx].size = vmmChunkSize;          // Virtual address space size
    vmmChunks[chunkIdx].physicalSize = physicalSize;  // Actual physical memory allocated
  }

  // Set access permissions for all GPUs - only for actually mapped memory regions
  int worldSize = bootNet.GetWorldSize();
  std::vector<hipMemAccessDesc> accessDescs(worldSize);

  for (int pe = 0; pe < worldSize; ++pe) {
    accessDescs[pe].location.type = hipMemLocationTypeDevice;
    accessDescs[pe].location.id = pe;
    accessDescs[pe].flags = hipMemAccessFlagsProtReadWrite;
  }

  // Set access permissions for each mapped physical memory region
  for (size_t i = 0; i < chunksNeeded; ++i) {
    size_t chunkIdx = startChunk + i;
    void* chunkPtr =
        static_cast<void*>(static_cast<char*>(vmmVirtualBasePtr) + chunkIdx * vmmChunkSize);
    size_t physicalSize = vmmChunks[chunkIdx].physicalSize;

    hipError_t result = hipMemSetAccess(chunkPtr, physicalSize, accessDescs.data(), worldSize);
    if (result != hipSuccess) {
      MORI_APP_WARN(
          "VMMAllocChunk failed: hipMemSetAccess failed for chunk {} at address {:p} with size {} "
          "bytes, hipError: {}",
          chunkIdx, chunkPtr, physicalSize, result);
      // Cleanup on access permission failure
      for (size_t j = 0; j <= i; ++j) {
        size_t cleanupIdx = startChunk + j;
        void* cleanupPtr =
            static_cast<void*>(static_cast<char*>(vmmVirtualBasePtr) + cleanupIdx * vmmChunkSize);
        HIP_RUNTIME_CHECK(hipMemUnmap(cleanupPtr, vmmChunks[cleanupIdx].physicalSize));
        HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[cleanupIdx].handle));
        vmmChunks[cleanupIdx].isAllocated = false;
        vmmChunks[cleanupIdx].physicalSize = 0;
      }
      return SymmMemObjPtr{nullptr, nullptr};
    }
  }

  // Create SymmMemObj for VMM allocation
  void* startPtr =
      static_cast<void*>(static_cast<char*>(vmmVirtualBasePtr) + startChunk * vmmChunkSize);
  return VMMRegisterSymmMemObj(startPtr, size, startChunk, chunksNeeded);
}

void SymmMemManager::VMMFreeChunk(void* localPtr) {
  std::lock_guard<std::mutex> lock(vmmLock);

  if (!vmmInitialized || !localPtr) {
    return;
  }

  // Find chunk index
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(vmmVirtualBasePtr);
  uintptr_t ptrAddr = reinterpret_cast<uintptr_t>(localPtr);

  if (ptrAddr < baseAddr || ptrAddr >= baseAddr + vmmVirtualSize) {
    return;  // Not in VMM range
  }

  size_t chunkIdx = (ptrAddr - baseAddr) / vmmChunkSize;

  // Find allocation size by checking registered object
  auto it = memObjPool.find(localPtr);
  if (it == memObjPool.end()) {
    return;
  }

  size_t allocSize = it->second.cpu->size;
  size_t chunksToFree = (allocSize + vmmChunkSize - 1) / vmmChunkSize;

  // Free chunks
  for (size_t i = 0; i < chunksToFree; ++i) {
    size_t idx = chunkIdx + i;
    if (idx < vmmMaxChunks && vmmChunks[idx].isAllocated) {
      void* chunkPtr =
          static_cast<void*>(static_cast<char*>(vmmVirtualBasePtr) + idx * vmmChunkSize);
      // Use the actual physical size that was mapped, not the full chunk size
      HIP_RUNTIME_CHECK(hipMemUnmap(chunkPtr, vmmChunks[idx].physicalSize));
      HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[idx].handle));
      vmmChunks[idx].isAllocated = false;
      vmmChunks[idx].size = 0;
      vmmChunks[idx].physicalSize = 0;
    }
  }

  HeapDeregisterSymmMemObj(localPtr);
}

SymmMemObjPtr SymmMemManager::VMMRegisterSymmMemObj(void* localPtr, size_t size, size_t startChunk,
                                                    size_t numChunks) {
  int worldSize = bootNet.GetWorldSize();
  int rank = bootNet.GetLocalRank();

  SymmMemObj* cpuMemObj = new SymmMemObj();
  cpuMemObj->localPtr = localPtr;
  cpuMemObj->size = size;

  // Calculate peer pointers based on VMM heap offsets
  cpuMemObj->peerPtrs = static_cast<uintptr_t*>(calloc(worldSize, sizeof(uintptr_t)));
  uintptr_t localOffset =
      reinterpret_cast<uintptr_t>(localPtr) - reinterpret_cast<uintptr_t>(vmmVirtualBasePtr);

  // Exchange shareable handles and create peer mappings
  std::vector<int> shareableHandles(worldSize * numChunks);
  
  // Collect local shareable handles
  for (size_t i = 0; i < numChunks; ++i) {
    size_t chunkIdx = startChunk + i;
    shareableHandles[rank * numChunks + i] = vmmChunks[chunkIdx].shareableHandle;
  }
  
  // Exchange shareable handles among all PEs
  bootNet.Allgather(shareableHandles.data() + rank * numChunks, 
                    shareableHandles.data(), 
                    sizeof(int) * numChunks);

  for (int i = 0; i < worldSize; ++i) {
    if (i == rank) {
      // Local case - use existing local address
      cpuMemObj->peerPtrs[i] = vmmHeapObj.cpu->peerPtrs[i] + localOffset;
    } else {
      // Remote case - import shareable handles and map to peer's virtual space
      cpuMemObj->peerPtrs[i] = vmmHeapObj.cpu->peerPtrs[i] + localOffset;
      
      // Import and map peer's memory chunks (for demonstration - actual implementation may vary)
      for (size_t j = 0; j < numChunks; ++j) {
        int peerShareableHandle = shareableHandles[i * numChunks + j];
        if (peerShareableHandle == -1) {
          MORI_APP_WARN("Skipping invalid shareable handle from PE {}, chunk {}", i, j);
          continue;
        }
        
        hipMemGenericAllocationHandle_t importedHandle;
        hipError_t result = hipMemImportFromShareableHandle(&importedHandle, 
                                                           &peerShareableHandle,
                                                           hipMemHandleTypePosixFileDescriptor);
        if (result != hipSuccess) {
          MORI_APP_WARN("Failed to import shareable handle from PE {}, chunk {}, hipError: {}", 
                        i, j, result);
          // Fallback to existing peer pointer mechanism
          continue;
        }
        
        // Note: In a full implementation, you would need to map this imported handle
        // to the peer's virtual address space. This requires careful coordination
        // of virtual address spaces across processes.
      }
    }
  }

  // VMM doesn't need IPC handles - access is managed through hipMemSetAccess and shareable handles
  cpuMemObj->ipcMemHandles =
      static_cast<hipIpcMemHandle_t*>(calloc(worldSize, sizeof(hipIpcMemHandle_t)));

  // Use VMM heap's unified RDMA keys instead of per-allocation registration
  cpuMemObj->peerRkeys = static_cast<uint32_t*>(calloc(worldSize, sizeof(uint32_t)));
  if (vmmHeapObj.IsValid() && vmmRdmaRegistered) {
    cpuMemObj->lkey = vmmHeapObj.cpu->lkey;
    memcpy(cpuMemObj->peerRkeys, vmmHeapObj.cpu->peerRkeys, sizeof(uint32_t) * worldSize);
  } else {
    cpuMemObj->lkey = 0;
    memset(cpuMemObj->peerRkeys, 0, sizeof(uint32_t) * worldSize);
  }

  SymmMemObj* gpuMemObj;
  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj, sizeof(SymmMemObj)));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj, cpuMemObj, sizeof(SymmMemObj), hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerPtrs, sizeof(uintptr_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerPtrs, cpuMemObj->peerPtrs,
                              sizeof(uintptr_t) * worldSize, hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerRkeys, sizeof(uint32_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerRkeys, cpuMemObj->peerRkeys,
                              sizeof(uint32_t) * worldSize, hipMemcpyHostToDevice));

  memObjPool.insert({localPtr, SymmMemObjPtr{cpuMemObj, gpuMemObj}});
  return memObjPool.at(localPtr);
}

bool SymmMemManager::VMMImportPeerMemory(int peerPe, void* localBaseAddr, size_t offset, size_t size, 
                                         const std::vector<int>& shareableHandles) {
  std::lock_guard<std::mutex> lock(vmmLock);
  
  if (!vmmInitialized) {
    MORI_APP_WARN("VMMImportPeerMemory failed: VMM heap not initialized");
    return false;
  }
  
  int worldSize = bootNet.GetWorldSize();
  if (peerPe >= worldSize || peerPe < 0) {
    MORI_APP_WARN("VMMImportPeerMemory failed: Invalid peerPe {}", peerPe);
    return false;
  }
  
  // Calculate target address in local virtual space
  void* targetAddr = static_cast<void*>(static_cast<char*>(localBaseAddr) + offset);
  size_t chunksNeeded = (size + vmmChunkSize - 1) / vmmChunkSize;
  
  // Import and map each chunk
  for (size_t i = 0; i < chunksNeeded && i < shareableHandles.size(); ++i) {
    hipMemGenericAllocationHandle_t importedHandle;
    
    // Import the shareable handle
    hipError_t result = hipMemImportFromShareableHandle(&importedHandle, 
                                                       (void *)&shareableHandles[i],
                                                       hipMemHandleTypePosixFileDescriptor);
    if (result != hipSuccess) {
      MORI_APP_WARN("VMMImportPeerMemory failed: hipMemImportFromShareableHandle failed for chunk {}, hipError: {}", 
                    i, result);
      
      // Cleanup already imported chunks
      for (size_t j = 0; j < i; ++j) {
        void* chunkAddr = static_cast<void*>(static_cast<char*>(targetAddr) + j * vmmChunkSize);
        hipError_t unmapResult = hipMemUnmap(chunkAddr, vmmChunkSize);
        if (unmapResult != hipSuccess) {
          MORI_APP_WARN("Failed to cleanup chunk {} during import failure, hipError: {}", j, unmapResult);
        }
      }
      return false;
    }
    
    // Map the imported handle to local virtual address space
    void* chunkAddr = static_cast<void*>(static_cast<char*>(targetAddr) + i * vmmChunkSize);
    size_t chunkSize = std::min(size - i * vmmChunkSize, vmmChunkSize);
    
    result = hipMemMap(chunkAddr, chunkSize, 0, importedHandle, 0);
    if (result != hipSuccess) {
      MORI_APP_WARN("VMMImportPeerMemory failed: hipMemMap failed for imported chunk {}, hipError: {}", 
                    i, result);
      
      // Release the imported handle and cleanup
      hipError_t releaseResult = hipMemRelease(importedHandle);
      if (releaseResult != hipSuccess) {
        MORI_APP_WARN("Failed to release imported handle, hipError: {}", releaseResult);
      }
      for (size_t j = 0; j < i; ++j) {
        void* prevChunkAddr = static_cast<void*>(static_cast<char*>(targetAddr) + j * vmmChunkSize);
        hipError_t unmapResult = hipMemUnmap(prevChunkAddr, vmmChunkSize);
        if (unmapResult != hipSuccess) {
          MORI_APP_WARN("Failed to cleanup chunk {} during map failure, hipError: {}", j, unmapResult);
        }
      }
      return false;
    }
    
    // Set access permissions
    hipMemAccessDesc accessDesc;
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = 0; // Current device
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    
    result = hipMemSetAccess(chunkAddr, chunkSize, &accessDesc, 1);
    if (result != hipSuccess) {
      MORI_APP_WARN("VMMImportPeerMemory warning: hipMemSetAccess failed for imported chunk {}, hipError: {}", 
                    i, result);
      // Continue without setting access - might still work in some cases
    }
  }
  
  MORI_APP_INFO("VMMImportPeerMemory: Successfully imported {} chunks from PE {} at offset {}", 
                chunksNeeded, peerPe, offset);
  return true;
}

}  // namespace application
}  // namespace mori
