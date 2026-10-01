/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

namespace {
#ifdef EMU_FAKE
  template<typename T, typename RedOp, typename Fan, int Direct, int P2p,
           int SlicePerChunk, int StepPerSlice, int Unroll, int MultimemSrcs, int MultimemDsts>
  __device__ __forceinline__
  Primitives<T, RedOp, Fan, Direct,
             ProtoSimple<SlicePerChunk, StepPerSlice, Unroll, MultimemSrcs, MultimemDsts>, P2p, false>
  makePrims(int tid, int nthreads, int const* recvPeers, int const* sendPeers,
            void const* inputBuf, void* outputBuf, uint64_t redOpArg,
            uint8_t group, uint8_t connIndexRecv, uint8_t connIndexSend,
            struct ncclDevWorkColl* work,
            ProtoSimple<SlicePerChunk, StepPerSlice, Unroll, MultimemSrcs, MultimemDsts>,
            int algo) {

    return Primitives<T, RedOp, Fan, Direct,
                      ProtoSimple<SlicePerChunk, StepPerSlice, Unroll, MultimemSrcs, MultimemDsts>, P2p, false>
      (tid, nthreads, recvPeers, sendPeers, inputBuf, outputBuf, redOpArg,
       group, connIndexRecv, connIndexSend, work,
       (struct ncclDevWorkP2p*)nullptr, 0, primsModeDefault, algo);
  }

  template<typename T, typename RedOp, typename Fan, int Direct, int P2p>
  __device__ __forceinline__
  Primitives<T, RedOp, Fan, Direct, ProtoLL, P2p, false>
  makePrims(int tid, int nthreads, int const* recvPeers, int const* sendPeers,
            void const* inputBuf, void* outputBuf, uint64_t redOpArg,
            uint8_t group, uint8_t connIndexRecv, uint8_t connIndexSend,
            struct ncclDevWorkColl* work,
            ProtoLL, int algo) {

    return Primitives<T, RedOp, Fan, Direct, ProtoLL, P2p, false>
      (tid, nthreads, recvPeers, sendPeers, inputBuf, outputBuf, redOpArg,
       group, connIndexRecv, connIndexSend, work,
       false, false, 0, algo);
  }

  template<typename T, typename RedOp, typename Fan, int Direct, int P2p>
  __device__ __forceinline__
  Primitives<T, RedOp, Fan, Direct, ProtoLL128, P2p, false>
  makePrims(int tid, int nthreads, int const* recvPeers, int const* sendPeers,
            void const* inputBuf, void* outputBuf, uint64_t redOpArg,
            uint8_t group, uint8_t connIndexRecv, uint8_t connIndexSend,
            struct ncclDevWorkColl* work,
            ProtoLL128, int algo) {

    return Primitives<T, RedOp, Fan, Direct, ProtoLL128, P2p, false>
      (tid, nthreads, recvPeers, sendPeers, inputBuf, outputBuf, redOpArg,
       group, connIndexRecv, connIndexSend, work,
       false, false, 0, algo);
  }
#endif

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    const int rank = ring->userRanks[0];
    const int nextRank = ring->userRanks[1];
    const int root = work->root;
    ssize_t chunkCount;
    ssize_t channelCount;
    ssize_t gridOffset;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;
    int workNthreads;
    bool isNetOffload = work->isOneRPN && work->netRegUsed;
#ifdef EMU_FAKE
    const int mode = ncclShmem.channel.ring.mode;
#endif

    T *inputBuf = (T*)work->sendbuff;
    T *outputBuf = (T*)work->recvbuff;
    workNthreads = isNetOffload ? WARP_SIZE : nthreads;

    if (tid < workNthreads) {
      // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
      // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
#ifndef EMU_FAKE
      Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0>
        prims(tid, workNthreads, &ring->prev, &ring->next, inputBuf, outputBuf, work->redOpArg, 0, 0, 0, work);
#else
      auto prims = makePrims<T, RedOp, FanSymmetric<1>, 1, 0>(
        tid, workNthreads, &ring->prev, &ring->next, inputBuf, outputBuf, work->redOpArg, 0, 0, 0, work, Proto{}, NCCL_ALGO_RING);
#endif

      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);

        if (rank == root) {
#ifndef EMU_FAKE
          if (inputBuf == outputBuf || isNetOffload) {
            prims.directSend(offset, offset, nelem);
          } else {
            prims.directCopySend(offset, offset, nelem);
          }
#else
          if (mode == SemiFake_Send) {
            if (inputBuf == outputBuf || isNetOffload) {
              prims.directSend(offset, offset, nelem);
            } else {
              prims.directCopySend(offset, offset, nelem);
            }
          }
#endif
        } else if (nextRank == root) {
#ifndef EMU_FAKE
          prims.directRecv(offset, nelem);
#else
          if (mode == SemiFake_Recv) {
            prims.directRecv(offset, nelem);
          }
#endif
        } else {
#ifndef EMU_FAKE
          prims.directRecvCopyDirectSend(offset, offset, nelem);
#else
          if (mode == SemiFake_Recv) {
            prims.directRecv(offset, nelem);
          }
          else if (mode == SemiFake_Send) {
            prims.directSend(offset, offset, nelem);
          }
          else if (mode == (SemiFake_Recv | SemiFake_Send)) {
            prims.directRecvCopyDirectSend(offset, offset, nelem);
          }
#endif
        }
      }
    } else if (inputBuf != outputBuf && rank == root) {
      inputBuf = inputBuf + gridOffset;
      outputBuf = outputBuf + gridOffset;
      reduceCopy<COLL_UNROLL, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>
        (tid - workNthreads, nthreads - workNthreads, work->redOpArg, &work->redOpArg, false, 1, (void**)&inputBuf, 1, (void**)&outputBuf, channelCount);
    }
    if (isNetOffload) barrier_sync(14, nthreads);
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};
