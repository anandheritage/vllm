# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for request queue implementations."""

import time

import pytest

from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    PriorityRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)

pytestmark = pytest.mark.cpu_test


class MockRequest:
    """Mock Request object for testing with proper comparison support."""

    def __init__(
        self,
        request_id: str,
        priority: int = 0,
        arrival_time: float | None = None,
    ):
        self.request_id = request_id
        self.priority = priority
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

    def __lt__(self, other: "MockRequest") -> bool:
        """Compare based on priority, arrival time, and request ID (same as Request)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)

    def __repr__(self) -> str:
        return f"MockRequest({self.request_id!r}, priority={self.priority})"


def create_mock_request(
    request_id: str,
    priority: int = 0,
    arrival_time: float | None = None,
) -> MockRequest:
    """Create a mock Request object for testing."""
    return MockRequest(request_id, priority, arrival_time)


class TestPriorityRequestQueue:
    """Tests for PriorityRequestQueue with lazy deletion."""

    def test_add_and_pop_request(self):
        """Test basic add and pop operations."""
        queue = PriorityRequestQueue()
        req1 = create_mock_request("req1", priority=1, arrival_time=1.0)
        req2 = create_mock_request("req2", priority=0, arrival_time=2.0)
        req3 = create_mock_request("req3", priority=1, arrival_time=0.5)

        queue.add_request(req1)
        queue.add_request(req2)
        queue.add_request(req3)

        assert len(queue) == 3

        # Should pop in priority order: req2 (priority=0), then req3 (priority=1,
        # earlier arrival), then req1 (priority=1, later arrival)
        assert queue.pop_request() == req2
        assert queue.pop_request() == req3
        assert queue.pop_request() == req1
        assert len(queue) == 0

    def test_peek_request(self):
        """Test peek operation returns top priority without removing."""
        queue = PriorityRequestQueue()
        req1 = create_mock_request("req1", priority=1, arrival_time=1.0)
        req2 = create_mock_request("req2", priority=0, arrival_time=2.0)

        queue.add_request(req1)
        queue.add_request(req2)

        # Peek should return req2 (highest priority) without removing
        assert queue.peek_request() == req2
        assert len(queue) == 2
        assert queue.peek_request() == req2  # Still there

    def test_remove_request_lazy_deletion(self):
        """Test that remove_request uses lazy deletion (O(1))."""
        queue = PriorityRequestQueue()
        requests = [
            create_mock_request(f"req{i}", priority=i, arrival_time=float(i))
            for i in range(10)
        ]

        for req in requests:
            queue.add_request(req)

        assert len(queue) == 10

        # Remove a request in the middle
        queue.remove_request(requests[5])
        assert len(queue) == 9

        # The deleted request should be tracked in _deleted counter
        assert queue._deleted[requests[5].request_id] == 1

        # Pop all remaining requests - should not include the removed one
        popped = [queue.pop_request() for _ in range(9)]
        assert requests[5] not in popped
        assert len(queue) == 0

    def test_remove_requests_lazy_deletion(self):
        """Test that remove_requests uses lazy deletion for multiple requests."""
        queue = PriorityRequestQueue()
        requests = [
            create_mock_request(f"req{i}", priority=i, arrival_time=float(i))
            for i in range(10)
        ]

        for req in requests:
            queue.add_request(req)

        # Remove multiple requests
        to_remove = [requests[2], requests[5], requests[8]]
        queue.remove_requests(to_remove)

        assert len(queue) == 7

        # Pop all remaining requests
        popped = [queue.pop_request() for _ in range(7)]
        for req in to_remove:
            assert req not in popped

    def test_remove_nonexistent_request(self):
        """Test removing a request that doesn't exist is a no-op."""
        queue = PriorityRequestQueue()
        req1 = create_mock_request("req1", priority=0)
        req2 = create_mock_request("req2", priority=1)

        queue.add_request(req1)

        # Remove a request that's not in the queue - should be a no-op
        # No tombstone is created for non-existent requests
        queue.remove_request(req2)

        # Length should still be 1 (no phantom tombstone created)
        assert len(queue) == 1
        assert queue._num_deleted == 0  # No tombstones created
        assert queue.pop_request() == req1
        assert len(queue) == 0

    def test_remove_nonexistent_does_not_affect_future_adds(self):
        """Test that removing non-existent request doesn't consume future adds."""
        queue = PriorityRequestQueue()
        req1 = create_mock_request("req1", priority=0)

        # Remove before adding - should be no-op
        queue.remove_request(req1)
        assert queue._num_deleted == 0

        # Now add the request
        queue.add_request(req1)
        assert len(queue) == 1

        # Should be able to pop it (no phantom tombstone consumed it)
        assert queue.pop_request() == req1
        assert len(queue) == 0

    def test_remove_by_request_id_not_identity(self):
        """Test that removal is by request_id, not object identity.

        This documents the semantic difference from FCFSRequestQueue which
        uses object identity for removal.
        """
        queue = PriorityRequestQueue()

        # Create two different objects with the same request_id
        req_a = create_mock_request("same_id", priority=0, arrival_time=1.0)
        req_b = create_mock_request("same_id", priority=0, arrival_time=2.0)

        # They are different objects
        assert req_a is not req_b

        # Add req_a to the queue
        queue.add_request(req_a)
        assert len(queue) == 1

        # Remove using req_b (different object, same request_id)
        # This should tombstone req_a because removal is ID-based
        queue.remove_request(req_b)

        # Queue should now be empty (req_a was tombstoned via req_b's ID)
        assert len(queue) == 0

    def test_pop_skips_deleted_items(self):
        """Test that pop correctly skips lazily deleted items."""
        queue = PriorityRequestQueue()
        req1 = create_mock_request("req1", priority=0, arrival_time=1.0)
        req2 = create_mock_request("req2", priority=0, arrival_time=2.0)
        req3 = create_mock_request("req3", priority=0, arrival_time=3.0)

        queue.add_request(req1)
        queue.add_request(req2)
        queue.add_request(req3)

        # Delete the first item (highest priority due to earliest arrival)
        queue.remove_request(req1)

        # Pop should skip req1 and return req2
        assert queue.pop_request() == req2
        assert queue.pop_request() == req3
        assert len(queue) == 0

    def test_peek_skips_deleted_items(self):
        """Test that peek correctly skips lazily deleted items."""
        queue = PriorityRequestQueue()
        req1 = create_mock_request("req1", priority=0, arrival_time=1.0)
        req2 = create_mock_request("req2", priority=0, arrival_time=2.0)

        queue.add_request(req1)
        queue.add_request(req2)

        # Delete the first item
        queue.remove_request(req1)

        # Peek should return req2
        assert queue.peek_request() == req2
        assert len(queue) == 1

    def test_readd_deleted_request(self):
        """Test that re-adding a deleted request works correctly."""
        queue = PriorityRequestQueue()
        req1 = create_mock_request("req1", priority=0, arrival_time=1.0)

        queue.add_request(req1)
        queue.remove_request(req1)
        assert len(queue) == 0

        # Re-add the same request
        queue.add_request(req1)
        assert len(queue) == 1
        assert queue.pop_request() == req1

    def test_multiple_add_remove_cycles(self):
        """Test multiple add/remove cycles with the same request."""
        queue = PriorityRequestQueue()
        req = create_mock_request("req", priority=0)

        queue.add_request(req)
        queue.remove_request(req)
        queue.add_request(req)
        queue.remove_request(req)
        queue.add_request(req)

        # Should have 1 valid item (heap=3, deleted=2, net=1)
        assert len(queue) == 1
        assert queue.pop_request() == req
        assert len(queue) == 0

    def test_iteration_skips_deleted(self):
        """Test that iteration skips deleted items."""
        queue = PriorityRequestQueue()
        requests = [
            create_mock_request(f"req{i}", priority=i, arrival_time=float(i))
            for i in range(5)
        ]

        for req in requests:
            queue.add_request(req)

        # Delete some requests
        queue.remove_request(requests[1])
        queue.remove_request(requests[3])

        # Iterate and collect
        iterated = list(queue)
        assert len(iterated) == 3
        assert requests[1] not in iterated
        assert requests[3] not in iterated

    def test_iteration_does_not_mutate_queue(self):
        """Test that iterating over the queue doesn't modify it."""
        queue = PriorityRequestQueue()
        requests = [
            create_mock_request(f"req{i}", priority=i, arrival_time=float(i))
            for i in range(5)
        ]

        for req in requests:
            queue.add_request(req)

        queue.remove_request(requests[2])

        # Record state before iteration
        len_before = len(queue)
        heap_size_before = len(queue._heap)
        deleted_before = dict(queue._deleted)

        # Iterate (should not mutate)
        _ = list(queue)

        # State should be unchanged
        assert len(queue) == len_before
        assert len(queue._heap) == heap_size_before
        assert dict(queue._deleted) == deleted_before

    def test_pop_with_many_consecutive_tombstones(self):
        """Test that pop handles many consecutive tombstones at the front.

        This tests the latency spike scenario where _skip_deleted must
        process many tombstones before returning a valid item.
        """
        queue = PriorityRequestQueue()

        # Add requests with same priority so arrival_time determines order
        requests = [
            create_mock_request(f"req{i}", priority=0, arrival_time=float(i))
            for i in range(100)
        ]

        for req in requests:
            queue.add_request(req)

        # Delete the first 90 requests (they're at the front due to arrival_time)
        for i in range(90):
            queue.remove_request(requests[i])

        assert len(queue) == 10

        # The next pop should skip 90 tombstones and return req90
        result = queue.pop_request()
        assert result == requests[90]

        # After pop, tombstones should be cleaned up
        assert queue._num_deleted == 0
        assert len(queue) == 9

    def test_bool_with_deleted_items(self):
        """Test __bool__ correctly accounts for deleted items."""
        queue = PriorityRequestQueue()
        req1 = create_mock_request("req1")

        assert not queue  # Empty queue is falsy

        queue.add_request(req1)
        assert queue  # Non-empty queue is truthy

        queue.remove_request(req1)
        assert not queue  # Queue with only deleted items is falsy

    def test_empty_queue_operations(self):
        """Test operations on empty queue."""
        queue = PriorityRequestQueue()

        assert len(queue) == 0
        assert not queue

        with pytest.raises(IndexError):
            queue.pop_request()

        with pytest.raises(IndexError):
            queue.peek_request()

    def test_compaction_triggered(self):
        """Test that heap compaction is triggered when threshold is exceeded."""
        queue = PriorityRequestQueue()

        # Create many requests
        num_requests = 300
        requests = [
            create_mock_request(f"req{i}", priority=i % 10, arrival_time=float(i))
            for i in range(num_requests)
        ]

        for req in requests:
            queue.add_request(req)

        # Delete more than threshold
        for i in range(200):
            queue.remove_request(requests[i])

        # After removing 200 out of 300, the heap should be compacted
        # when the threshold is exceeded (200/300 > 0.5 and 200 > 100)
        assert len(queue) == 100

        # The compaction should have cleared the _deleted counter
        assert queue._num_deleted < 200

    def test_compaction_preserves_duplicate_request_ids(self):
        """Test that compaction correctly handles multiple adds of same request.

        If request "A" is added 3 times and removed 1 time, compaction should
        only remove 1 entry, not all 3 (which was a bug in earlier versions).
        """
        queue = PriorityRequestQueue()

        # Create a request and add it multiple times
        req = create_mock_request("same_id", priority=0, arrival_time=1.0)

        # Add the same request 3 times
        queue.add_request(req)
        queue.add_request(req)
        queue.add_request(req)
        assert len(queue) == 3
        assert queue._in_heap["same_id"] == 3

        # Remove once
        queue.remove_request(req)
        assert len(queue) == 2
        assert queue._deleted["same_id"] == 1

        # Force compaction by adding/removing many items to trigger threshold
        for i in range(150):
            r = create_mock_request(f"filler{i}", priority=1, arrival_time=float(i))
            queue.add_request(r)
            queue.remove_request(r)

        # After compaction, we should still have 2 copies of "same_id"
        # (not 0, which would happen if compaction dropped all matching IDs)
        same_id_count = sum(1 for r in queue if r.request_id == "same_id")
        assert same_id_count == 2

    def test_prepend_request(self):
        """Test prepend_request adds according to priority (same as add)."""
        queue = PriorityRequestQueue()
        req1 = create_mock_request("req1", priority=1)
        req2 = create_mock_request("req2", priority=0)

        queue.prepend_request(req1)
        queue.prepend_request(req2)

        # Should still be ordered by priority
        assert queue.pop_request() == req2

    def test_prepend_requests(self):
        """Test prepend_requests adds all according to priority."""
        queue1 = PriorityRequestQueue()
        queue2 = PriorityRequestQueue()

        req1 = create_mock_request("req1", priority=2)
        req2 = create_mock_request("req2", priority=0)
        req3 = create_mock_request("req3", priority=1)

        queue2.add_request(req2)
        queue2.add_request(req3)

        queue1.add_request(req1)
        queue1.prepend_requests(queue2)

        assert len(queue1) == 3
        assert queue1.pop_request() == req2  # Lowest priority value


class TestFCFSRequestQueue:
    """Tests for FCFSRequestQueue."""

    def test_add_and_pop_request(self):
        """Test FCFS ordering."""
        queue = FCFSRequestQueue()
        req1 = create_mock_request("req1")
        req2 = create_mock_request("req2")
        req3 = create_mock_request("req3")

        queue.add_request(req1)
        queue.add_request(req2)
        queue.add_request(req3)

        # Should pop in FIFO order
        assert queue.pop_request() == req1
        assert queue.pop_request() == req2
        assert queue.pop_request() == req3

    def test_remove_requests(self):
        """Test removing multiple requests."""
        queue = FCFSRequestQueue()
        requests = [create_mock_request(f"req{i}") for i in range(5)]

        for req in requests:
            queue.add_request(req)

        queue.remove_requests([requests[1], requests[3]])

        assert len(queue) == 3
        remaining = list(queue)
        assert requests[1] not in remaining
        assert requests[3] not in remaining


class TestCreateRequestQueue:
    """Tests for the factory function."""

    def test_create_priority_queue(self):
        """Test creating a priority queue."""
        queue = create_request_queue(SchedulingPolicy.PRIORITY)
        assert isinstance(queue, PriorityRequestQueue)

    def test_create_fcfs_queue(self):
        """Test creating a FCFS queue."""
        queue = create_request_queue(SchedulingPolicy.FCFS)
        assert isinstance(queue, FCFSRequestQueue)
