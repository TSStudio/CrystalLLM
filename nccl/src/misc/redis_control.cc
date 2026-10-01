#ifdef EMU_MOCK

#include "redis_control.h"

/*************************************************************************
 * Thread-local persistent connection
 *
 * Each pthread gets its own redisContext via __thread. This avoids mutex
 * contention on the hot path while keeping connections alive across calls.
 * hiredis redisContext is NOT thread-safe, so one-per-thread is the
 * simplest correct design.
 *
 * Environment: CRYSTAL_REDIS_HOST (default 127.0.0.1)
 *              CRYSTAL_REDIS_PORT (default 6379)
 ************************************************************************/

static __thread redisContext* tls_redis = NULL;

static redisContext* getConn(void) {
  if (tls_redis && !tls_redis->err) return tls_redis;
  // Stale/broken connection — clean up first
  if (tls_redis) { redisFree(tls_redis); tls_redis = NULL; }

  const char* host = ncclGetEnv("CRYSTAL_REDIS_HOST");
  const char* portEnv = ncclGetEnv("CRYSTAL_REDIS_PORT");
  if (!host) host = "127.0.0.1";
  if (!portEnv) portEnv = "6379";
  int port = atoi(portEnv);

  struct timeval tv; tv.tv_sec = 1; tv.tv_usec = 0; // 1s connect timeout
  tls_redis = redisConnectWithTimeout(host, port, tv);
  if (tls_redis == NULL || tls_redis->err) {
    if (tls_redis) {
      INFO(NCCL_INIT, "Redis connect error: %s", tls_redis->errstr);
      redisFree(tls_redis);
      tls_redis = NULL;
    }
    return NULL;
  }
  // Set command timeout (100ms) for ongoing operations
  struct timeval cmdTv; cmdTv.tv_sec = 0; cmdTv.tv_usec = 100000;
  redisSetTimeout(tls_redis, cmdTv);
  return tls_redis;
}

// Mark connection as broken so next getConn() reconnects.
static void markBroken(void) {
  if (tls_redis) { redisFree(tls_redis); tls_redis = NULL; }
}

/*************************************************************************
 * Internal helpers
 ************************************************************************/

static int parseTruthy(const char* s, size_t len) {
  if (len == 0 || s == NULL) return 0;
  if (len == 1 && s[0] == '1') return 1;
  if (strncasecmp(s, "true", len) == 0) return 1;
  if (strncasecmp(s, "yes", len) == 0) return 1;
  if (strncasecmp(s, "on", len) == 0) return 1;
  return 0;
}

// Parse a REDIS_REPLY_ARRAY of strings into a malloc'd int array.
// Caller must free *out on success.
static ncclResult_t parseIntList(redisReply* reply, int** out, int* outCount) {
  if (reply == NULL || reply->type != REDIS_REPLY_ARRAY) return ncclInternalError;
  int count = (int)reply->elements;
  if (count <= 0) {
    *out = NULL;
    *outCount = 0;
    return ncclSuccess;
  }
  int* arr = NULL;
  if (ncclCalloc(&arr, count) != ncclSuccess) return ncclSystemError;
  for (int i = 0; i < count; i++) {
    if (reply->element[i]->type != REDIS_REPLY_STRING) {
      free(arr);
      return ncclInternalError;
    }
    arr[i] = atoi(reply->element[i]->str);
  }
  *out = arr;
  *outCount = count;
  return ncclSuccess;
}

/*************************************************************************
 * Public API (ordered as in redis_control.h)
 ************************************************************************/

ncclResult_t ncclRedisControlAddSandboxNeighbor(const char* key, int neighborRank) {
  if (key == NULL) return ncclInvalidArgument;
  redisContext* c = getConn();
  if (!c) return ncclSystemError;

  // LPOS to avoid duplicates
  redisReply* pos = (redisReply*)redisCommand(c, "LPOS %s %d", key, neighborRank);
  if (pos == NULL) { markBroken(); return ncclSystemError; }
  if (pos->type == REDIS_REPLY_INTEGER && pos->integer >= 0) {
    freeReplyObject(pos);
    return ncclSuccess; // already exists
  }
  freeReplyObject(pos);

  redisReply* rep = (redisReply*)redisCommand(c, "LPUSH %s %d", key, neighborRank);
  if (rep == NULL) { markBroken(); return ncclSystemError; }
  freeReplyObject(rep);
  return ncclSuccess;
}

ncclResult_t ncclRedisControlSetInt(const char* key, int value) {
  if (key == NULL) return ncclInvalidArgument;
  redisContext* c = getConn();
  if (!c) return ncclSystemError;

  redisReply* reply = (redisReply*)redisCommand(c, "SET %s %d", key, value);
  if (reply == NULL) { markBroken(); return ncclSystemError; }
  freeReplyObject(reply);
  return ncclSuccess;
}

ncclResult_t ncclRedisControlGetInt(const char* key, int* outVal) {
  if (key == NULL || outVal == NULL) return ncclInvalidArgument;
  redisContext* c = getConn();
  if (!c) return ncclSystemError;

  redisReply* reply = (redisReply*)redisCommand(c, "GET %s", key);
  if (reply == NULL) { markBroken(); return ncclSystemError; }

  ncclResult_t ret = ncclSuccess;
  if (reply->type == REDIS_REPLY_STRING) {
    *outVal = atoi(reply->str);
  } else if (reply->type == REDIS_REPLY_INTEGER) {
    *outVal = (int)reply->integer;
  } else if (reply->type == REDIS_REPLY_NIL) {
    ret = ncclInvalidUsage;
  } else {
    ret = ncclInternalError;
  }
  freeReplyObject(reply);
  return ret;
}

ncclResult_t ncclRedisControlGetTruthy(const char* key, int* out) {
  if (key == NULL || out == NULL) return ncclInvalidArgument;
  redisContext* c = getConn();
  if (!c) return ncclSystemError;

  redisReply* reply = (redisReply*)redisCommand(c, "GET %s", key);
  if (reply == NULL) { markBroken(); return ncclSystemError; }

  ncclResult_t ret = ncclSuccess;
  if (reply->type == REDIS_REPLY_STRING) {
    *out = parseTruthy(reply->str, (size_t)reply->len) ? 1 : 0;
  } else if (reply->type == REDIS_REPLY_NIL) {
    ret = ncclInvalidUsage;
  } else {
    ret = ncclInternalError;
  }
  freeReplyObject(reply);
  return ret;
}

ncclResult_t ncclRedisControlGetIntList(const char* key, int** out, int* outCount) {
  if (key == NULL || out == NULL || outCount == NULL) return ncclInvalidArgument;
  *out = NULL;
  *outCount = 0;
  redisContext* c = getConn();
  if (!c) return ncclSystemError;

  redisReply* reply = (redisReply*)redisCommand(c, "LRANGE %s 0 -1", key);
  if (reply == NULL) { markBroken(); return ncclSystemError; }
  ncclResult_t ret = parseIntList(reply, out, outCount);
  freeReplyObject(reply);
  return ret;
}

ncclResult_t ncclRedisControlSetBlob(const char* key, const void* data, size_t len) {
  if (key == NULL || data == NULL) return ncclInvalidArgument;
  redisContext* c = getConn();
  if (!c) return ncclSystemError;

  redisReply* reply = (redisReply*)redisCommand(c, "SET %s %b", key, data, (size_t)len);
  if (reply == NULL) { markBroken(); return ncclSystemError; }
  freeReplyObject(reply);
  return ncclSuccess;
}

ncclResult_t ncclRedisControlGetBlob(const char* key, void** outData, size_t* outLen) {
  if (key == NULL || outData == NULL || outLen == NULL) return ncclInvalidArgument;
  *outData = NULL; *outLen = 0;
  redisContext* c = getConn();
  if (!c) return ncclSystemError;

  redisReply* reply = (redisReply*)redisCommand(c, "GET %s", key);
  if (reply == NULL) { markBroken(); return ncclSystemError; }

  ncclResult_t ret = ncclSuccess;
  if (reply->type == REDIS_REPLY_STRING) {
    char* buf = NULL;
    if (ncclCalloc(&buf, (size_t)reply->len) != ncclSuccess) { ret = ncclSystemError; }
    else {
      memcpy(buf, reply->str, (size_t)reply->len);
      *outData = (void*)buf;
      *outLen = (size_t)reply->len;
    }
  } else if (reply->type == REDIS_REPLY_NIL) {
    ret = ncclInvalidUsage;
  } else {
    ret = ncclInternalError;
  }
  freeReplyObject(reply);
  return ret;
}

ncclResult_t ncclRedisControlResolveCommHash(uint64_t commHash, char* outGroupDesc, size_t outLen) {
  if (outGroupDesc == NULL || outLen == 0) return ncclInvalidArgument;
  outGroupDesc[0] = '\0';
  redisContext* c = getConn();
  if (!c) return ncclSystemError;

  char redisKey[MAX_REDIS_CONTROL_KEY_LEN];
  snprintf(redisKey, sizeof(redisKey), "comm:%016llx:groupDesc", (unsigned long long)commHash);

  redisReply* reply = (redisReply*)redisCommand(c, "GET %s", redisKey);
  if (reply == NULL) { markBroken(); return ncclSystemError; }

  ncclResult_t ret = ncclSuccess;
  if (reply->type == REDIS_REPLY_STRING) {
    size_t copyLen = (size_t)reply->len < outLen - 1 ? (size_t)reply->len : outLen - 1;
    memcpy(outGroupDesc, reply->str, copyLen);
    outGroupDesc[copyLen] = '\0';
  } else if (reply->type == REDIS_REPLY_NIL) {
    ret = ncclInvalidUsage;
  } else {
    ret = ncclInternalError;
  }
  freeReplyObject(reply);
  return ret;
}

ncclResult_t ncclRedisControlPublishGroupDesc(uint64_t commHash, const char* groupDesc) {
  if (groupDesc == NULL || groupDesc[0] == '\0') return ncclInvalidArgument;
  redisContext* c = getConn();
  if (!c) return ncclSystemError;

  char redisKey[MAX_REDIS_CONTROL_KEY_LEN];
  snprintf(redisKey, sizeof(redisKey), "comm:%016llx:groupDesc", (unsigned long long)commHash);

  redisReply* reply = (redisReply*)redisCommand(c, "SET %s %b", redisKey, groupDesc, strlen(groupDesc));
  if (reply == NULL) { markBroken(); return ncclSystemError; }
  freeReplyObject(reply);
  return ncclSuccess;
}

#endif /* EMU_MOCK */
