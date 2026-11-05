"""Conformance suite for session storage backends (``hermes_state_port``).

Core runs it against the local backend (``test_local.py``). A backend plugin runs the same suite
by importing :class:`StorageConformance` into its own test module and supplying a ``backend``
fixture that returns a :class:`Backend` for its storage::

    from tests.conformance.storage.suite import Backend, StorageConformance

    class TestMyBackend(StorageConformance):
        @pytest.fixture
        def backend(self, tmp_path):
            yield Backend(...)

A test that needs a service it cannot reach fails, never skips.
"""
