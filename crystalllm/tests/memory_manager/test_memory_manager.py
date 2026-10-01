# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Unit tests for crystalllm.replay.memory_manager (CUDA-independent parts).

These tests exercise pure key-encoding logic and the Lua-script-driven Redis
primitives (acquire_if_ready / claim_for_release). They require a local Redis
server reachable at localhost:6379 (the repo's dev environment ships one),
but never touch CUDA/GPU state.
"""

import unittest

import redis

from crystalllm.replay import memory_manager as mm


def _redis_available() -> bool:
    try:
        client = redis.Redis(host="localhost", port=6379, db=mm.MEMORY_MANAGER_REDIS_DB)
        client.ping()
        return True
    except redis.exceptions.ConnectionError:
        return False


REDIS_AVAILABLE = _redis_available()


class TestDemandIdEncoding(unittest.TestCase):
    def test_round_trip_basic(self):
        demand_id = mm.encode_demand_id(
            3, "all_reduce", "all_reduce(tensor)", mm.MEMORY_TENSOR_INPUT, [4, 8], "torch.float16"
        )
        decoded = mm.decode_demand_id(demand_id)
        self.assertEqual(
            decoded, (3, "all_reduce", "all_reduce(tensor)", mm.MEMORY_TENSOR_INPUT, [4, 8], "torch.float16")
        )

    def test_round_trip_empty_shape(self):
        demand_id = mm.encode_demand_id(0, "barrier", "barrier()", mm.MEMORY_TENSOR_INPUT, [], "torch.float32")
        decoded = mm.decode_demand_id(demand_id)
        self.assertEqual(decoded[4], [])

    def test_round_trip_special_characters_in_signature(self):
        signature = 'isend(tensor, dst=1, tag=0):[a, b]{c:d} 中文 "quoted"'
        demand_id = mm.encode_demand_id(1, "isend", signature, mm.MEMORY_TENSOR_INPUT, [1024], "torch.bfloat16")
        decoded = mm.decode_demand_id(demand_id)
        self.assertEqual(decoded[2], signature)

    def test_demand_id_is_url_safe(self):
        signature = "op(x)/y+z=w"
        demand_id = mm.encode_demand_id(7, "irecv", signature, mm.MEMORY_TENSOR_OUTPUT, [2, 2], "torch.int64")
        # base64.urlsafe alphabet only, no padding.
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
        self.assertTrue(set(demand_id) <= allowed)
        self.assertNotIn("=", demand_id)

    def test_different_ranks_produce_different_demand_ids(self):
        a = mm.encode_demand_id(0, "all_reduce", "sig", mm.MEMORY_TENSOR_INPUT, [4], "torch.float32")
        b = mm.encode_demand_id(1, "all_reduce", "sig", mm.MEMORY_TENSOR_INPUT, [4], "torch.float32")
        self.assertNotEqual(a, b)

    def test_direction_affects_demand_id(self):
        a = mm.encode_demand_id(0, "irecv", "sig", mm.MEMORY_TENSOR_INPUT, [4], "torch.float32")
        b = mm.encode_demand_id(0, "irecv", "sig", mm.MEMORY_TENSOR_OUTPUT, [4], "torch.float32")
        self.assertNotEqual(a, b)


class TestShareKeyEncoding(unittest.TestCase):
    def test_round_trip(self):
        share_key = mm.encode_share_key(
            "all_gather_into_tensor", "sig", mm.MEMORY_TENSOR_OUTPUT, [16, 16], "torch.float32"
        )
        decoded = mm.decode_share_key(share_key)
        self.assertEqual(decoded, ("all_gather_into_tensor", "sig", mm.MEMORY_TENSOR_OUTPUT, [16, 16], "torch.float32"))

    def test_share_key_ignores_rank(self):
        demand_a = mm.encode_demand_id(0, "broadcast", "sig", mm.MEMORY_TENSOR_INPUT, [4], "torch.float32")
        demand_b = mm.encode_demand_id(1, "broadcast", "sig", mm.MEMORY_TENSOR_INPUT, [4], "torch.float32")
        _, func_a, sig_a, dir_a, shape_a, dtype_a = mm.decode_demand_id(demand_a)
        _, func_b, sig_b, dir_b, shape_b, dtype_b = mm.decode_demand_id(demand_b)
        share_a = mm.encode_share_key(func_a, sig_a, dir_a, shape_a, dtype_a)
        share_b = mm.encode_share_key(func_b, sig_b, dir_b, shape_b, dtype_b)
        self.assertEqual(share_a, share_b)

    def test_demand_id_from_share_key(self):
        share_key = mm.encode_share_key("all_reduce", "sig", mm.MEMORY_TENSOR_INPUT, [8], "torch.float16")
        demand_id = mm.demand_id_from_share_key(share_key, rank=5)
        self.assertEqual(
            demand_id, mm.encode_demand_id(5, "all_reduce", "sig", mm.MEMORY_TENSOR_INPUT, [8], "torch.float16")
        )


class TestKeyBuilders(unittest.TestCase):
    def test_all_keys_carry_local_rank_prefix(self):
        local_rank = 2
        demand_id = mm.encode_demand_id(0, "all_reduce", "sig", mm.MEMORY_TENSOR_INPUT, [4], "torch.float32")
        request_id = mm.make_request_id(0, 1)
        keys = [
            mm.key_init_done(local_rank),
            mm.key_init_managed_ranks(local_rank),
            mm.key_init_reserved_memory_ratio(local_rank),
            mm.key_demand_level(local_rank, demand_id),
            mm.key_demand_target_ref(local_rank, demand_id),
            mm.key_pool_anything(local_rank, "torch.float32"),
            mm.key_allocation(local_rank, "shared", 1),
            mm.key_instant_next_use_index(local_rank, demand_id),
            mm.key_instant_total_use_count(local_rank, demand_id),
            mm.key_instant_use_allocation_ref(local_rank, demand_id, 0),
            mm.key_request_queue(local_rank),
            mm.key_request_demand_id(local_rank, request_id),
            mm.key_rank_events_channel(local_rank, 3),
        ]
        prefix = f"memory:{local_rank}:"
        for key in keys:
            self.assertTrue(key.startswith(prefix), key)

    def test_allocation_key_format(self):
        self.assertEqual(mm.key_allocation(0, "shared", 1), "memory:0:allocation:shared:000001")
        self.assertEqual(mm.key_allocation(0, "persistent", 18), "memory:0:allocation:persistent:000018")
        self.assertEqual(mm.key_allocation(0, "instant", 77), "memory:0:allocation:instant:000077")

    def test_request_id_round_trip_rank(self):
        request_id = mm.make_request_id(5, 42)
        self.assertEqual(request_id, "r5_42")
        self.assertEqual(mm.parse_request_id_rank(request_id), 5)

    def test_field_helper(self):
        self.assertEqual(
            mm.redis_field("memory:0:allocation:shared:000001", "state"), "memory:0:allocation:shared:000001:state"
        )


@unittest.skipUnless(REDIS_AVAILABLE, "requires a local redis-server on localhost:6379")
class TestAcquireIfReadyLua(unittest.TestCase):
    def setUp(self):
        self.local_rank = 0
        self.client = mm.MemoryManagerRedisClient(self.local_rank)
        self.client.command.flushdb()
        self.target_ref = mm.key_allocation(self.local_rank, "persistent", 1)

    def tearDown(self):
        self.client.command.flushdb()

    def _set_resource(self, state: str, version: int = 0, user_count: int = 0):
        self.client.command.set(mm.redis_field(self.target_ref, "state"), state)
        self.client.command.set(mm.redis_field(self.target_ref, "version"), version)
        self.client.command.set(mm.redis_field(self.target_ref, "user_count"), user_count)

    def test_not_ready_when_evicted(self):
        self._set_resource(mm.STATE_EVICTED)
        result = self.client.acquire_if_ready(self.target_ref)
        self.assertEqual(result, mm.ACQUIRE_NOT_READY)
        self.assertEqual(int(self.client.command.get(mm.redis_field(self.target_ref, "user_count"))), 0)

    def test_not_ready_when_evicting(self):
        self._set_resource(mm.STATE_EVICTING)
        result = self.client.acquire_if_ready(self.target_ref)
        self.assertEqual(result, mm.ACQUIRE_NOT_READY)

    def test_ready_without_expected_version_increments_count(self):
        self._set_resource(mm.STATE_READY, version=0, user_count=0)
        result = self.client.acquire_if_ready(self.target_ref)
        self.assertEqual(result, mm.ACQUIRE_READY)
        self.assertEqual(int(self.client.command.get(mm.redis_field(self.target_ref, "user_count"))), 1)

    def test_ready_with_matching_expected_version_increments_count(self):
        self._set_resource(mm.STATE_READY, version=3, user_count=2)
        result = self.client.acquire_if_ready(self.target_ref, expected_version=3)
        self.assertEqual(result, mm.ACQUIRE_READY)
        self.assertEqual(int(self.client.command.get(mm.redis_field(self.target_ref, "user_count"))), 3)

    def test_version_mismatch_does_not_increment_count(self):
        self._set_resource(mm.STATE_READY, version=3, user_count=2)
        result = self.client.acquire_if_ready(self.target_ref, expected_version=2)
        self.assertEqual(result, mm.ACQUIRE_VERSION_MISMATCH)
        self.assertEqual(int(self.client.command.get(mm.redis_field(self.target_ref, "user_count"))), 2)

    def test_repeated_acquire_accumulates_count(self):
        self._set_resource(mm.STATE_READY, version=0, user_count=0)
        for expected in (1, 2, 3):
            result = self.client.acquire_if_ready(self.target_ref)
            self.assertEqual(result, mm.ACQUIRE_READY)
            self.assertEqual(int(self.client.command.get(mm.redis_field(self.target_ref, "user_count"))), expected)


@unittest.skipUnless(REDIS_AVAILABLE, "requires a local redis-server on localhost:6379")
class TestClaimForReleaseLua(unittest.TestCase):
    def setUp(self):
        self.local_rank = 0
        self.client = mm.MemoryManagerRedisClient(self.local_rank)
        self.client.command.flushdb()
        self.target_ref = mm.key_allocation(self.local_rank, "shared", 2)

    def tearDown(self):
        self.client.command.flushdb()

    def _set_resource(self, state: str, user_count: int):
        self.client.command.set(mm.redis_field(self.target_ref, "state"), state)
        self.client.command.set(mm.redis_field(self.target_ref, "user_count"), user_count)

    def test_claim_succeeds_when_ready_and_unused(self):
        self._set_resource(mm.STATE_READY, user_count=0)
        result = self.client.claim_for_release(self.target_ref, mm.STATE_EVICTING)
        self.assertEqual(result, mm.CLAIM_CLAIMED)
        self.assertEqual(self.client.command.get(mm.redis_field(self.target_ref, "state")).decode(), mm.STATE_EVICTING)

    def test_claim_skipped_when_in_use(self):
        self._set_resource(mm.STATE_READY, user_count=1)
        result = self.client.claim_for_release(self.target_ref, mm.STATE_EVICTING)
        self.assertEqual(result, mm.CLAIM_SKIP)
        self.assertEqual(self.client.command.get(mm.redis_field(self.target_ref, "state")).decode(), mm.STATE_READY)

    def test_claim_skipped_when_not_ready(self):
        self._set_resource(mm.STATE_EVICTED, user_count=0)
        result = self.client.claim_for_release(self.target_ref, mm.STATE_EVICTING)
        self.assertEqual(result, mm.CLAIM_SKIP)

    def test_claim_for_destroying_instant(self):
        self._set_resource(mm.STATE_READY, user_count=0)
        result = self.client.claim_for_release(self.target_ref, mm.STATE_DESTROYING)
        self.assertEqual(result, mm.CLAIM_CLAIMED)
        self.assertEqual(
            self.client.command.get(mm.redis_field(self.target_ref, "state")).decode(), mm.STATE_DESTROYING
        )

    def test_race_only_one_claim_wins(self):
        # Simulate two eviction attempts racing on the same resource: since Lua
        # scripts execute atomically, only the first call transitions state; the
        # second must observe the already-transitioned (non-READY) state and skip.
        self._set_resource(mm.STATE_READY, user_count=0)
        first = self.client.claim_for_release(self.target_ref, mm.STATE_EVICTING)
        second = self.client.claim_for_release(self.target_ref, mm.STATE_EVICTING)
        self.assertEqual(first, mm.CLAIM_CLAIMED)
        self.assertEqual(second, mm.CLAIM_SKIP)


@unittest.skipUnless(REDIS_AVAILABLE, "requires a local redis-server on localhost:6379")
class TestPoolAcquireAlwaysReady(unittest.TestCase):
    """ANYTHING pools use the same acquire_if_ready path but are always READY."""

    def test_pool_state_key_reused_by_acquire(self):
        local_rank = 0
        client = mm.MemoryManagerRedisClient(local_rank)
        client.command.flushdb()
        pool_ref = mm.key_pool_anything(local_rank, "torch.float32")
        client.command.set(mm.redis_field(pool_ref, "state"), mm.STATE_READY)
        client.command.set(mm.redis_field(pool_ref, "version"), 0)
        client.command.set(mm.redis_field(pool_ref, "user_count"), 0)

        result = client.acquire_if_ready(pool_ref)
        self.assertEqual(result, mm.ACQUIRE_READY)
        self.assertEqual(int(client.command.get(mm.redis_field(pool_ref, "user_count"))), 1)
        client.command.flushdb()


if __name__ == "__main__":
    unittest.main()
