#ifndef NCCL_REDIS_CONTROL_H_
#define NCCL_REDIS_CONTROL_H_

#ifdef EMU_MOCK

#include "utils.h"
#include <hiredis/hiredis.h>

#define MAX_REDIS_CONTROL_KEY_LEN 512

// Maximum size for a groupDesc buffer.
// Must be strictly smaller than MAX_REDIS_CONTROL_KEY_LEN so that callers can
// safely format keys like "group:<groupDesc>:<suffix>" into a buffer of size
// MAX_REDIS_CONTROL_KEY_LEN without triggering -Wformat-truncation.
#define MAX_GROUP_DESC_LEN 256

/*************************************************************************
 * Redis control helpers
 *
 * Redis address is configured via CRYSTAL_REDIS_HOST / CRYSTAL_REDIS_PORT
 * environment variables (defaults: 127.0.0.1:6379).
 * Connections are managed internally (thread-local, auto-connect on
 * first use, auto-reconnect on failure).
 *
 * All keys are assembled by the caller; no default keys are hardcoded.
 ************************************************************************/

// Append an int to a Redis LIST if not already present (LPOS + LPUSH).
ncclResult_t ncclRedisControlAddSandboxNeighbor(const char* key, int neighborRank);

// Integer read/write.
ncclResult_t ncclRedisControlSetInt(const char* key, int value);
ncclResult_t ncclRedisControlGetInt(const char* key, int* outVal);

// Read a truthy string value ("1"/"true"/"yes"/"on") → *out=1, else 0.
// Returns ncclInvalidUsage if key does not exist.
ncclResult_t ncclRedisControlGetTruthy(const char* key, int* out);

// Read a Redis LIST as a malloc'd int array. Caller owns *out (must free).
ncclResult_t ncclRedisControlGetIntList(const char* key, int** out, int* outCount);

// Binary-safe blob read/write. Caller owns *outData on GetBlob (must free).
ncclResult_t ncclRedisControlSetBlob(const char* key, const void* data, size_t len);
ncclResult_t ncclRedisControlGetBlob(const char* key, void** outData, size_t* outLen);

// Top-level entry: read config from Redis, run mockInitTransportsRank,
// compute sandbox neighbors, publish ag3 data + neighbors to Redis.
// groupDesc must be non-NULL and non-empty.
__attribute__((visibility("default")))
ncclResult_t ncclRedisControlSetNeighbors(const char* groupDesc);

// Resolve a commHash to its groupDesc string via Redis.
// Key format: comm:<commHash_hex_16>:groupDesc
// Writes result into caller-provided buffer outGroupDesc[outLen].
// Returns ncclInvalidUsage if the key does not exist.
ncclResult_t ncclRedisControlResolveCommHash(uint64_t commHash, char* outGroupDesc, size_t outLen);

// Publish a commHash → groupDesc mapping. Symmetric to the resolver above.
// Used by ncclCommInitRankFunc to expose comm->config.commName to the mock
// bootstrap proxy (which keys per-group ag3 data on groupDesc).
ncclResult_t ncclRedisControlPublishGroupDesc(uint64_t commHash, const char* groupDesc);

#endif /* EMU_MOCK */

#endif /* NCCL_REDIS_CONTROL_H_ */
