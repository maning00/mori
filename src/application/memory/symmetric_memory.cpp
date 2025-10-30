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

  int worldSize = bootNet.GetWorldSize();
  int myPe = bootNet.GetLocalRank();

  std::vector<TransportType> transportTypes = context.GetTransportTypes();
  // Calculate per-PE virtual address space size
  vmmPerPeerSize = virtualSize;
  size_t p2pPeCount = 0;
  for (int pe = 0; pe < worldSize; ++pe) {
    if (transportTypes[pe] == TransportType::P2P && myPe != pe) {
      p2pPeCount++;
    }
  }

  // Only allocate virtual space for P2P accessible PEs
  size_t totalVirtualSize = vmmPerPeerSize * (p2pPeCount + 1);

  vmmVirtualSize = virtualSize;  // Keep original size for local PE
  vmmChunkSize = chunkSize;
  vmmMaxChunks = virtualSize / chunkSize;

  MORI_APP_INFO(
      "VMM Heap Initialization: virtualSize = {}, chunkSize = {}, minChunkSize = {}, maxChunks = "
      "{}, "
      "worldSize = {}, p2pPeCount = {}, perPeerSize = {}, totalVirtualSize = {}",
      vmmVirtualSize, vmmChunkSize, vmmMinChunkSize, vmmMaxChunks, worldSize, p2pPeCount,
      vmmPerPeerSize, totalVirtualSize);

  // Reserve virtual address space for all PEs
  hipError_t result = hipMemAddressReserve(&vmmVirtualBasePtr, totalVirtualSize, 0, nullptr, 0);
  if (result != hipSuccess) {
    MORI_APP_WARN(
        "InitializeVMMHeap failed: hipMemAddressReserve failed for total size {}, hipError: {}",
        totalVirtualSize, result);
    return false;
  }

  // Set up peer base pointers for each PE
  vmmPeerBasePtrs.resize(worldSize);
  size_t virtualOffset = 0;
  for (int i = 0; i < worldSize; ++i) {
    int pe = (myPe + i) % worldSize;

    if (pe == myPe || transportTypes[pe] == TransportType::P2P) {
      vmmPeerBasePtrs[pe] =
          static_cast<void*>(static_cast<char*>(vmmVirtualBasePtr) + virtualOffset);
      virtualOffset += vmmPerPeerSize;
    } else {
      vmmPeerBasePtrs[pe] = nullptr;
    }
  }

  // Initialize chunk tracking (only for local PE initially)
  vmmChunks.resize(vmmMaxChunks);

  // Initialize VMM heap unified RDMA registration for the entire virtual space
  uint32_t vmmHeapLkey{0};
  std::vector<uint32_t> vmmHeapRkeys;
  vmmHeapRkeys.resize(worldSize, 0);
  RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
  if (rdmaDeviceContext) {
    application::RdmaMemoryRegion mr =
        rdmaDeviceContext->RegisterRdmaMemoryRegion(vmmVirtualBasePtr, virtualSize);
    vmmHeapLkey = mr.lkey;
    vmmHeapRkeys[myPe] = mr.rkey;
    vmmRdmaRegistered = true;
  }

  // Exchange RDMA keys among all PEs
  if (vmmRdmaRegistered) {
    bootNet.Allgather(&vmmHeapRkeys[myPe], vmmHeapRkeys.data(), sizeof(uint32_t));
  }

  // Create SymmMemObjPtr for the entire VMM heap
  SymmMemObj* cpuHeapObj = new SymmMemObj();
  cpuHeapObj->localPtr = vmmVirtualBasePtr;
  cpuHeapObj->size = virtualSize;

  // Exchange virtual base pointers among all PEs
  cpuHeapObj->peerPtrs = static_cast<uintptr_t*>(calloc(worldSize, sizeof(uintptr_t)));
  bootNet.Allgather(&vmmVirtualBasePtr, cpuHeapObj->peerPtrs, sizeof(uintptr_t));
  for (int pe = 0; pe < worldSize; ++pe) {
    if (vmmPeerBasePtrs[pe] != nullptr) {
      cpuHeapObj->peerPtrs[pe] = reinterpret_cast<uintptr_t>(vmmPeerBasePtrs[pe]);
    }
  }

  // VMM doesn't need IPC handles - access is managed through hipMemSetAccess and shareable handles
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

  int rank = bootNet.GetLocalRank();

  // Deregister VMM heap's unified RDMA registration before cleanup
  if (vmmRdmaRegistered) {
    RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
    if (rdmaDeviceContext) {
      rdmaDeviceContext->DeregisterRdmaMemoryRegion(vmmVirtualBasePtr);
    }
    vmmRdmaRegistered = false;
  }

  // Free all allocated chunks in local PE's virtual address space
  for (size_t i = 0; i < vmmMaxChunks; ++i) {
    if (vmmChunks[i].isAllocated) {
      void* chunkPtr =
          static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) + i * vmmChunkSize);
      // Use the actual physical size that was mapped
      HIP_RUNTIME_CHECK(hipMemUnmap(chunkPtr, vmmChunks[i].physicalSize));
      HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[i].handle));
    }
  }

  // Free virtual address space (entire multi-PE space)
  if (vmmVirtualBasePtr) {
    size_t totalVirtualSize = vmmPerPeerSize * bootNet.GetWorldSize();
    HIP_RUNTIME_CHECK(hipMemAddressFree(vmmVirtualBasePtr, totalVirtualSize));
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
  vmmPeerBasePtrs.clear();
  vmmMinChunkSize = 0;
  vmmPerPeerSize = 0;
  vmmInitialized = false;
}

SymmMemObjPtr SymmMemManager::VMMAllocChunk(size_t size, uint32_t allocType) {
  std::lock_guard<std::mutex> lock(vmmLock);

  if (!vmmInitialized) {
    MORI_APP_WARN("VMMAllocChunk failed: VMM heap not initialized");
    return SymmMemObjPtr{nullptr, nullptr};
  }

  size_t chunksNeeded = (size + vmmChunkSize - 1) / vmmChunkSize;

  // Find contiguous free chunks in local PE's address space
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
  int worldSize = bootNet.GetWorldSize();
  int rank = bootNet.GetLocalRank();

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

  // Store shareable handles for exchange - only the allocating PE creates handles
  std::vector<int> localShareableHandles(chunksNeeded, -1);

  // Only rank 0 creates the physical memory and exports shareable handles
  if (rank == 0) {
    for (size_t i = 0; i < chunksNeeded; ++i) {
      size_t chunkIdx = startChunk + i;
      void* localChunkPtr =
          static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) + chunkIdx * vmmChunkSize);

      // Calculate actual physical memory needed for this chunk
      size_t remainingSize = size - (i * vmmChunkSize);
      size_t physicalSize = std::max(std::min(remainingSize, vmmChunkSize), vmmMinChunkSize);

      // Create physical memory with actual needed size
      result = hipMemCreate(&vmmChunks[chunkIdx].handle, physicalSize, &allocProp, 0);
      if (result != hipSuccess) {
        MORI_APP_WARN(
            "VMMAllocChunk failed: hipMemCreate failed for chunk {} with size {} bytes, allocType: "
            "{}, device: {}, hipError: {}",
            chunkIdx, physicalSize, allocType, currentDev, result);
        // Cleanup already allocated chunks
        for (size_t j = 0; j < i; ++j) {
          size_t cleanupIdx = startChunk + j;
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

      localShareableHandles[i] = vmmChunks[chunkIdx].shareableHandle;
      MORI_APP_TRACE(
          "VMMAllocChunk: RANK {} (allocator) Created chunk {} with physical size {} bytes, shareable handle {} (PID: {})",
          rank, chunkIdx, physicalSize, vmmChunks[chunkIdx].shareableHandle, getpid());
      
      // Map physical memory to local virtual address
      result = hipMemMap(localChunkPtr, physicalSize, 0, vmmChunks[chunkIdx].handle, 0);
      if (result != hipSuccess) {
        MORI_APP_WARN(
            "VMMAllocChunk failed: hipMemMap failed for chunk {} at address {:p} with size {} bytes, "
            "hipError: {}",
            chunkIdx, localChunkPtr, physicalSize, result);
        HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[chunkIdx].handle));
        // Cleanup already allocated chunks
        for (size_t j = 0; j < i; ++j) {
          size_t cleanupIdx = startChunk + j;
          void* cleanupPtr = static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) +
                                                cleanupIdx * vmmChunkSize);
          HIP_RUNTIME_CHECK(hipMemUnmap(cleanupPtr, vmmChunks[cleanupIdx].physicalSize));
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
  } else {
    // Non-allocating PEs just prepare empty handles - they will import from rank 0
    MORI_APP_TRACE("VMMAllocChunk: RANK {} (consumer) waiting for shareable handles from rank 0", rank);
  }

  // Exchange shareable handles among all PEs using bootstrap network
  // Use a flat array for proper Allgather communication
  std::vector<int> allShareableHandlesFlat(worldSize * chunksNeeded, -1);

  // Copy local shareable handles to the correct position in flat array
  for (size_t i = 0; i < chunksNeeded; ++i) {
    allShareableHandlesFlat[rank * chunksNeeded + i] = localShareableHandles[i];
  }

  // Exchange via bootstrap network - each PE contributes chunksNeeded handles
  bootNet.Allgather(localShareableHandles.data(), allShareableHandlesFlat.data(),
                    sizeof(int) * chunksNeeded);

  MORI_APP_TRACE("RANK {} VMMAllocChunk: Exchanged shareable handles among all PEs", rank);

  // Print all shareable handles for debugging
  for (int pe = 0; pe < worldSize; ++pe) {
    for (size_t i = 0; i < chunksNeeded; ++i) {
      int handleValue = allShareableHandlesFlat[pe * chunksNeeded + i];
      MORI_APP_TRACE("RANK {} VMMAllocChunk: PE {} created shareable handle[{}] = {} (PID would be different per PE)", 
                     rank, pe, i, handleValue);
    }
  }
  // Import and map to peer virtual address spaces for cross-process access
  for (int pe = 0; pe < worldSize; ++pe) {
    if (pe == rank) continue;  // Skip self - no need to import our own handles
    
    // Only map to P2P accessible PEs
    MORI_APP_TRACE(
        "VMMAllocChunk: RANK {} Checking transport type for PE {} context.GetTransportType(pe) = "
        "{}",
        rank, pe, context.GetTransportType(pe));
    if (context.GetTransportType(pe) == TransportType::P2P) {
      for (size_t i = 0; i < chunksNeeded; ++i) {
        // Always import from rank 0's shareable handles (the memory allocator)
        int handleValue = allShareableHandlesFlat[0 * chunksNeeded + i];
        MORI_APP_TRACE(
            "VMMAllocChunk: RANK {} Importing rank 0's shareable handle for chunk {} to map to PE {} space, handle = {}",
            rank, i, pe, handleValue);
        if (handleValue == -1) {
          MORI_APP_WARN("Skipping invalid shareable handle from rank 0, chunk {}", i);
          continue;
        }

        // Calculate target address in PE's virtual space
        void* peerChunkPtr = static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[pe]) +
                                                (startChunk + i) * vmmChunkSize);
        
        MORI_APP_TRACE(
            "VMMAllocChunk: RANK {} Mapping rank 0's chunk {} to PE {} virtual address {:p}",
            rank, i, pe, peerChunkPtr);
            
        // Import the shareable handle from rank 0
        hipMemGenericAllocationHandle_t importedHandle;
        result = hipMemImportFromShareableHandle(&importedHandle, (void*)&handleValue,
                                                 hipMemHandleTypePosixFileDescriptor);
        if (result != hipSuccess) {
          MORI_APP_WARN(
              "Failed to import shareable handle to PE {} virtual space, chunk {}, hipError: {}",
              pe, i, result);
          continue;
        }
        MORI_APP_TRACE(
            "VMMAllocChunk: RANK {} Imported rank 0's shareable handle for chunk {} to map to PE {} virtual "
            "address {:p}",
            rank, i, pe, peerChunkPtr);
        
        // Map to PE's virtual address space
        size_t chunkIdx = startChunk + i;
        size_t physicalSize;
        
        // Calculate expected physical size (same logic as rank 0 used)
        size_t remainingSize = size - (i * vmmChunkSize);
        physicalSize = std::max(std::min(remainingSize, vmmChunkSize), vmmMinChunkSize);
        
        MORI_APP_TRACE("VMMAllocChunk: RANK {} calculated physicalSize = {} for chunk {}", 
                       rank, physicalSize, i);
        
        result = hipMemMap(peerChunkPtr, physicalSize, 0, importedHandle, 0);
        if (result != hipSuccess) {
          MORI_APP_WARN(
              "Failed to map imported memory to PE {} virtual space, chunk {}, hipError: {}", pe, i,
              result);
          HIP_RUNTIME_CHECK(hipMemRelease(importedHandle));
          continue;
        }

        // Set access permissions for this virtual mapping
        hipMemAccessDesc accessDesc;
        accessDesc.location.type = hipMemLocationTypeDevice;
        accessDesc.location.id = pe;
        accessDesc.flags = hipMemAccessFlagsProtReadWrite;

        result = hipMemSetAccess(peerChunkPtr, physicalSize, &accessDesc, 1);
        if (result != hipSuccess) {
          MORI_APP_WARN("Failed to set access for PE {} virtual space, chunk {}, hipError: {}", pe,
                        i, result);
        }
      }
    }
  }
  
  // Handle self-mapping for non-rank 0 PEs (import rank 0's memory to own space)
  if (rank != 0) {
    for (size_t i = 0; i < chunksNeeded; ++i) {
      int handleValue = allShareableHandlesFlat[0 * chunksNeeded + i];
      if (handleValue == -1) {
        MORI_APP_WARN("RANK {} cannot import invalid handle from rank 0 for self-mapping", rank);
        continue;
      }
      
      void* localChunkPtr = static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) +
                                              (startChunk + i) * vmmChunkSize);
      
      MORI_APP_TRACE("VMMAllocChunk: RANK {} importing rank 0's handle {} for self-mapping to {:p}",
                     rank, handleValue, localChunkPtr);
      
      // Import the shareable handle from rank 0
      hipMemGenericAllocationHandle_t importedHandle;
      result = hipMemImportFromShareableHandle(&importedHandle, (void*)&handleValue,
                                               hipMemHandleTypePosixFileDescriptor);
      if (result != hipSuccess) {
        MORI_APP_WARN(
            "RANK {} failed to import shareable handle for self-mapping, chunk {}, hipError: {}",
            rank, i, result);
        continue;
      }
      
      // Calculate physical size
      size_t remainingSize = size - (i * vmmChunkSize);
      size_t physicalSize = std::max(std::min(remainingSize, vmmChunkSize), vmmMinChunkSize);
      
      // Map to own virtual address space
      result = hipMemMap(localChunkPtr, physicalSize, 0, importedHandle, 0);
      if (result != hipSuccess) {
        MORI_APP_WARN(
            "RANK {} failed to map imported memory for self-mapping, chunk {}, hipError: {}", 
            rank, i, result);
        HIP_RUNTIME_CHECK(hipMemRelease(importedHandle));
        continue;
      }
      
      // Update chunk tracking for non-rank 0 PE
      size_t chunkIdx = startChunk + i;
      vmmChunks[chunkIdx].isAllocated = true;
      vmmChunks[chunkIdx].size = vmmChunkSize;
      vmmChunks[chunkIdx].physicalSize = physicalSize;
      vmmChunks[chunkIdx].handle = importedHandle;  // Store the imported handle
      vmmChunks[chunkIdx].shareableHandle = -1;     // We don't export, rank 0 does
    }
  }
  MORI_APP_TRACE("VMMAllocChunk: RANK {} Allocated {} bytes using {} chunks starting at chunk {}",
                 rank, size, chunksNeeded, startChunk);
  // Set access permissions for local mapped memory regions
  std::vector<hipMemAccessDesc> accessDescs(worldSize);
  for (int pe = 0; pe < worldSize; ++pe) {
    accessDescs[pe].location.type = hipMemLocationTypeDevice;
    accessDescs[pe].location.id = pe;
    accessDescs[pe].flags = hipMemAccessFlagsProtReadWrite;
  }

  // Set access permissions for each local mapped physical memory region
  for (size_t i = 0; i < chunksNeeded; ++i) {
    size_t chunkIdx = startChunk + i;
    void* chunkPtr =
        static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) + chunkIdx * vmmChunkSize);
    size_t physicalSize = vmmChunks[chunkIdx].physicalSize;

    hipError_t result = hipMemSetAccess(chunkPtr, physicalSize, accessDescs.data(), worldSize);
    if (result != hipSuccess) {
      MORI_APP_WARN(
          "VMMAllocChunk warning: hipMemSetAccess failed for local chunk {} at address {:p} with "
          "size {} "
          "bytes, hipError: {}",
          chunkIdx, chunkPtr, physicalSize, result);
    }
  }

  // Create SymmMemObj for VMM allocation
  void* startPtr =
      static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) + startChunk * vmmChunkSize);
  MORI_APP_TRACE("VMMAllocChunk: Allocated memory at virtual address {:p} of size {} bytes",
                 startPtr, size);
  return VMMRegisterSymmMemObj(startPtr, size, startChunk, chunksNeeded);
}

void SymmMemManager::VMMFreeChunk(void* localPtr) {
  std::lock_guard<std::mutex> lock(vmmLock);

  if (!vmmInitialized || !localPtr) {
    return;
  }

  int rank = bootNet.GetLocalRank();
  int worldSize = bootNet.GetWorldSize();

  // Find chunk index in local PE's virtual address space
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(vmmPeerBasePtrs[rank]);
  uintptr_t ptrAddr = reinterpret_cast<uintptr_t>(localPtr);

  if (ptrAddr < baseAddr || ptrAddr >= baseAddr + vmmPerPeerSize) {
    return;  // Not in local PE's VMM range
  }

  size_t chunkIdx = (ptrAddr - baseAddr) / vmmChunkSize;

  // Find allocation size by checking registered object
  auto it = memObjPool.find(localPtr);
  if (it == memObjPool.end()) {
    return;
  }

  size_t allocSize = it->second.cpu->size;
  size_t chunksToFree = (allocSize + vmmChunkSize - 1) / vmmChunkSize;

  // Free chunks from local PE's virtual address space
  for (size_t i = 0; i < chunksToFree; ++i) {
    size_t idx = chunkIdx + i;
    if (idx < vmmMaxChunks && vmmChunks[idx].isAllocated) {
      void* chunkPtr =
          static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) + idx * vmmChunkSize);
      // Use the actual physical size that was mapped, not the full chunk size
      HIP_RUNTIME_CHECK(hipMemUnmap(chunkPtr, vmmChunks[idx].physicalSize));
      HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[idx].handle));
      vmmChunks[idx].isAllocated = false;
      vmmChunks[idx].size = 0;
      vmmChunks[idx].physicalSize = 0;
      vmmChunks[idx].shareableHandle = -1;
    }
  }

  // Also unmap from peer virtual address spaces for P2P accessible PEs
  for (int pe = 0; pe < worldSize; ++pe) {
    if (pe == rank) continue;  // Skip self, already unmapped above

    // Only unmap from P2P accessible PEs where we previously mapped
    if (context.GetTransportType(pe) == TransportType::P2P && vmmPeerBasePtrs[pe] != nullptr) {
      for (size_t i = 0; i < chunksToFree; ++i) {
        size_t idx = chunkIdx + i;
        if (idx < vmmMaxChunks) {
          void* peerChunkPtr =
              static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[pe]) + idx * vmmChunkSize);

          // Get the physical size that was originally mapped
          size_t physicalSize = vmmChunks[idx].physicalSize;
          if (physicalSize > 0) {
            hipError_t result = hipMemUnmap(peerChunkPtr, physicalSize);
            if (result != hipSuccess) {
              MORI_APP_WARN("Failed to unmap peer memory for PE {} chunk {}, hipError: {}", pe, idx,
                            result);
            }
          }
        }
      }
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

  // Calculate peer pointers based on VMM per-PE virtual address spaces
  cpuMemObj->peerPtrs = static_cast<uintptr_t*>(calloc(worldSize, sizeof(uintptr_t)));

  uintptr_t localOffset =
      reinterpret_cast<uintptr_t>(localPtr) - reinterpret_cast<uintptr_t>(vmmVirtualBasePtr);
  MORI_APP_TRACE("VMMRegisterSymmMemObj: localOffset = {}", localOffset);
  // Set peer pointers to corresponding addresses in each PE's virtual address space
  for (int pe = 0; pe < worldSize; ++pe) {
    cpuMemObj->peerPtrs[pe] = vmmHeapObj.cpu->peerPtrs[pe] + localOffset;
    MORI_APP_TRACE("VMMRegisterSymmMemObj: peerPtrs[{}] = {:p}", pe,
                   reinterpret_cast<void*>(cpuMemObj->peerPtrs[pe]));
  }
  MORI_APP_TRACE(
      "VMMRegisterSymmMemObj: Registered memory at local address {:p} of size {} "
      "bytes",
      localPtr, size);
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
  MORI_APP_TRACE("VMMRegisterSymmMemObj: Using lkey {} for RDMA access", cpuMemObj->lkey);
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
  MORI_APP_TRACE(
      "VMMRegisterSymmMemObj: rank: {} Registered memory at local address {:p} of size {} "
      "bytes",
      rank, localPtr, size);
  return memObjPool.at(localPtr);
}

bool SymmMemManager::VMMImportPeerMemory(int peerPe, void* localBaseAddr, size_t offset,
                                         size_t size, const std::vector<int>& shareableHandles) {
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

  // Calculate target address in peer's dedicated virtual space
  void* targetAddr = static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[peerPe]) + offset);
  size_t chunksNeeded = (size + vmmChunkSize - 1) / vmmChunkSize;

  MORI_APP_INFO(
      "VMMImportPeerMemory: Importing {} chunks from PE {} to peer virtual space at offset {}",
      chunksNeeded, peerPe, offset);

  // Import and map each chunk to peer's virtual address space
  for (size_t i = 0; i < chunksNeeded && i < shareableHandles.size(); ++i) {
    if (shareableHandles[i] == -1) {
      MORI_APP_WARN("VMMImportPeerMemory: Invalid shareable handle for chunk {}", i);
      continue;
    }

    hipMemGenericAllocationHandle_t importedHandle;

    // Import the shareable handle
    hipError_t result = hipMemImportFromShareableHandle(
        &importedHandle, (void*)&shareableHandles[i], hipMemHandleTypePosixFileDescriptor);
    if (result != hipSuccess) {
      MORI_APP_WARN(
          "VMMImportPeerMemory failed: hipMemImportFromShareableHandle failed for chunk {}, "
          "hipError: {}",
          i, result);

      // Cleanup already imported chunks
      for (size_t j = 0; j < i; ++j) {
        void* chunkAddr = static_cast<void*>(static_cast<char*>(targetAddr) + j * vmmChunkSize);
        hipError_t unmapResult = hipMemUnmap(chunkAddr, vmmChunkSize);
        if (unmapResult != hipSuccess) {
          MORI_APP_WARN("Failed to cleanup chunk {} during import failure, hipError: {}", j,
                        unmapResult);
        }
      }
      return false;
    }

    // Map the imported handle to peer's virtual address space
    void* chunkAddr = static_cast<void*>(static_cast<char*>(targetAddr) + i * vmmChunkSize);
    size_t chunkSize = std::min(size - i * vmmChunkSize, vmmChunkSize);

    result = hipMemMap(chunkAddr, chunkSize, 0, importedHandle, 0);
    if (result != hipSuccess) {
      MORI_APP_WARN(
          "VMMImportPeerMemory failed: hipMemMap failed for imported chunk {}, hipError: {}", i,
          result);

      // Release the imported handle and cleanup
      hipError_t releaseResult = hipMemRelease(importedHandle);
      if (releaseResult != hipSuccess) {
        MORI_APP_WARN("Failed to release imported handle, hipError: {}", releaseResult);
      }
      for (size_t j = 0; j < i; ++j) {
        void* prevChunkAddr = static_cast<void*>(static_cast<char*>(targetAddr) + j * vmmChunkSize);
        hipError_t unmapResult = hipMemUnmap(prevChunkAddr, vmmChunkSize);
        if (unmapResult != hipSuccess) {
          MORI_APP_WARN("Failed to cleanup chunk {} during map failure, hipError: {}", j,
                        unmapResult);
        }
      }
      return false;
    }

    // Set access permissions for the current device to access the imported memory
    hipMemAccessDesc accessDesc;
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = 0;  // Current device
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;

    result = hipMemSetAccess(chunkAddr, chunkSize, &accessDesc, 1);
    if (result != hipSuccess) {
      MORI_APP_WARN(
          "VMMImportPeerMemory warning: hipMemSetAccess failed for imported chunk {}, hipError: {}",
          i, result);
      // Continue without setting access - might still work in some cases
    }
  }

  MORI_APP_INFO(
      "VMMImportPeerMemory: Successfully imported {} chunks from PE {} to peer virtual space",
      chunksNeeded, peerPe);
  return true;
}

}  // namespace application
}  // namespace mori
