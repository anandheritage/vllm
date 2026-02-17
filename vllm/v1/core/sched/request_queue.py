# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations with lazy deletion.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.

    This implementation uses lazy deletion for O(1) remove operations instead
    of O(n) heap rebuilds. Deleted request IDs are tracked with a counter,
    and items are skipped during pop/peek operations. The heap is compacted
    when the number of tombstones exceeds a threshold.
    """

    # Compact the heap when deleted items exceed this fraction of heap size
    _COMPACTION_THRESHOLD = 0.5
    # Minimum number of deleted items before considering compaction
    _MIN_DELETED_FOR_COMPACTION = 100

    def __init__(self) -> None:
        self._heap: list[Request] = []
        # Counter of tombstones per request_id (supports multiple deletions
        # of the same request_id if it was added multiple times)
        self._deleted: Counter[str] = Counter()
        # Counter tracking how many instances of each request_id are in heap
        # This prevents phantom tombstones from consuming future requests
        self._in_heap: Counter[str] = Counter()
        # Total number of tombstones in the heap
        self._num_deleted: int = 0

    def _maybe_compact(self) -> None:
        """Compact the heap if too many deleted items have accumulated.

        This maintains heap efficiency by removing tombstones when they
        exceed a threshold fraction of the heap size.
        """
        if self._num_deleted < self._MIN_DELETED_FOR_COMPACTION:
            return
        if len(self._heap) == 0:
            self._deleted.clear()
            self._num_deleted = 0
            return
        if self._num_deleted / len(self._heap) > self._COMPACTION_THRESHOLD:
            # Use counter-decrementing to correctly handle multiple instances
            # of the same request_id (only skip tombstoned entries, keep rest)
            deleted_copy = self._deleted.copy()
            new_heap: list[Request] = []
            for r in self._heap:
                if deleted_copy[r.request_id] > 0:
                    deleted_copy[r.request_id] -= 1
                    self._in_heap[r.request_id] -= 1
                else:
                    new_heap.append(r)
            self._heap = new_heap
            heapq.heapify(self._heap)
            self._deleted.clear()
            self._num_deleted = 0
            # Clean up zero/negative entries in _in_heap.
            # Unary + on Counter returns a new Counter with only positive counts.
            # See: https://docs.python.org/3/library/collections.html#collections.Counter
            self._in_heap = +self._in_heap

    def _skip_deleted(self) -> None:
        """Remove deleted items from the front of the heap.

        This is called before peek/pop to ensure we return a valid item.
        """
        while self._heap and self._deleted.get(self._heap[0].request_id, 0) > 0:
            request = heapq.heappop(self._heap)
            self._deleted[request.request_id] -= 1
            self._num_deleted -= 1
            self._in_heap[request.request_id] -= 1
            # Clean up zero entries to save memory
            if self._deleted[request.request_id] == 0:
                del self._deleted[request.request_id]
            if self._in_heap[request.request_id] == 0:
                del self._in_heap[request.request_id]

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)
        self._in_heap[request.request_id] += 1

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        self._skip_deleted()
        if not self._heap:
            raise IndexError("pop from empty heap")
        request = heapq.heappop(self._heap)
        self._in_heap[request.request_id] -= 1
        if self._in_heap[request.request_id] == 0:
            del self._in_heap[request.request_id]
        return request

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        self._skip_deleted()
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue.

        This is O(1) using lazy deletion - the request is marked as deleted
        and will be skipped during pop/peek operations.

        Important semantic notes:
        - Removal is by request_id, not object identity. If multiple Request
          objects share the same request_id (which shouldn't happen in normal
          usage), this will tombstone one matching entry regardless of which
          object is passed.
        - Unlike the original O(n) implementation that raised ValueError
          for missing requests, this is idempotent - removing a request that
          doesn't exist in the queue is a safe no-op.
        - This differs from FCFSRequestQueue which uses object identity.
        """
        # Only create tombstone if there are un-tombstoned entries in heap
        live_count = (
            self._in_heap[request.request_id] - self._deleted[request.request_id]
        )
        if live_count > 0:
            self._deleted[request.request_id] += 1
            self._num_deleted += 1
            self._maybe_compact()

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue.

        This is O(k) where k is the number of requests to remove,
        using lazy deletion.

        Note: Removing requests that don't exist in the queue is a no-op.
        """
        for request in requests:
            live_count = (
                self._in_heap[request.request_id] - self._deleted[request.request_id]
            )
            if live_count > 0:
                self._deleted[request.request_id] += 1
                self._num_deleted += 1
        self._maybe_compact()

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue.

        This returns the exact count of live (non-tombstoned) items.
        """
        return len(self._heap) - self._num_deleted

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        # Create a copy of the heap for iteration, excluding deleted items
        # Use a local counter copy to track which items to skip
        deleted_copy = self._deleted.copy()
        heap_copy: list[Request] = []
        for r in self._heap:
            if deleted_copy[r.request_id] > 0:
                deleted_copy[r.request_id] -= 1
            else:
                heap_copy.append(r)
        heapq.heapify(heap_copy)
        while heap_copy:
            yield heapq.heappop(heap_copy)


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
