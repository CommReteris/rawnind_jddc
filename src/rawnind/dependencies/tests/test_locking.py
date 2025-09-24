import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
from rawnind.dependencies import locking


def test_lock_unlock_basic_cpu(tmp_path):
    """
    Test basic lock acquisition and release for CPU resource.

    Objective: Verify that lock() and unlock() work correctly for CPU locking.
    Test criteria: Lock file should be created when locking and removed when unlocking.
    How testing for this criteria fulfills purpose: Ensures basic locking mechanism works without file system issues.
    What components are mocked, monkeypatched, or are fixtures: tmp_path fixture for isolated directory.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Uses real file operations in temporary directory to test actual behavior.
    """
    # Override global variables for test isolation
    locking.LOCKDIR = str(tmp_path)
    locking.LOCK_FPATH = {"cpu": os.path.join(locking.LOCKDIR, "cpu"), "gpu": os.path.join(locking.LOCKDIR, "gpu")}

    # Test basic lock acquisition and release for CPU
    locking.lock("cpu")
    assert (tmp_path / "cpu").exists()
    locking.unlock("cpu")
    assert not (tmp_path / "cpu").exists()


def test_lock_unlock_basic_gpu(tmp_path):
    """
    Test basic lock acquisition and release for GPU resource.

    Objective: Verify that lock() and unlock() work correctly for GPU locking.
    Test criteria: Lock file should be created when locking and removed when unlocking.
    How testing for this criteria fulfills purpose: Ensures basic locking mechanism works for GPU resources.
    What components are mocked, monkeypatched, or are fixtures: tmp_path fixture for isolated directory.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Uses real file operations in temporary directory to test actual behavior.
    """
    # Override global variables for test isolation
    locking.LOCKDIR = str(tmp_path)
    locking.LOCK_FPATH = {"cpu": os.path.join(locking.LOCKDIR, "cpu"), "gpu": os.path.join(locking.LOCKDIR, "gpu")}

    # Test basic lock acquisition and release for GPU
    locking.lock("gpu")
    assert (tmp_path / "gpu").exists()
    locking.unlock("gpu")
    assert not (tmp_path / "gpu").exists()


def test_is_locked_not_locked(tmp_path):
    """
    Test is_locked() when resource is not locked.

    Objective: Verify is_locked() returns False when no lock file exists.
    Test criteria: Function returns False for non-existent lock file.
    How testing for this criteria fulfills purpose: Confirms correct behavior when resource is available.
    What components are mocked, monkeypatched, or are fixtures: tmp_path fixture for isolated directory.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Tests actual file existence check.
    """
    locking.LOCKDIR = str(tmp_path)
    locking.LOCK_FPATH = {"cpu": os.path.join(locking.LOCKDIR, "cpu"), "gpu": os.path.join(locking.LOCKDIR, "gpu")}

    assert not locking.is_locked("cpu")
    assert not locking.is_locked("gpu")


def test_is_locked_own_lock(tmp_path):
    """
    Test is_locked() when current process owns the lock.

    Objective: Verify is_locked() returns False when current process owns the lock.
    Test criteria: Function returns False when lock file contains current PID.
    How testing for this criteria fulfills purpose: Ensures process doesn't block itself.
    What components are mocked, monkeypatched, or are fixtures: tmp_path fixture for isolated directory.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Tests actual PID comparison logic.
    """
    locking.LOCKDIR = str(tmp_path)
    locking.LOCK_FPATH = {"cpu": os.path.join(locking.LOCKDIR, "cpu"), "gpu": os.path.join(locking.LOCKDIR, "gpu")}

    locking.lock("cpu")
    assert not locking.is_locked("cpu")  # Should return False when we own the lock


def test_is_locked_other_process(tmp_path):
    """
    Test is_locked() when another process owns the lock.

    Objective: Verify is_locked() returns True when another active process owns the lock.
    Test criteria: Function returns True when lock file contains different PID.
    How testing for this criteria fulfills purpose: Ensures proper blocking for concurrent processes.
    What components are mocked, monkeypatched, or are fixtures: tmp_path fixture for isolated directory, mocked os.kill.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Simulates real process existence check.
    """
    locking.LOCKDIR = str(tmp_path)
    locking.LOCK_FPATH = {"cpu": os.path.join(locking.LOCKDIR, "cpu"), "gpu": os.path.join(locking.LOCKDIR, "gpu")}

    # Create lock file with different PID
    lock_file = tmp_path / "cpu"
    lock_file.write_text("99999\n")  # Different PID

    with patch('os.kill') as mock_kill:
        mock_kill.return_value = None  # Process exists
        assert locking.is_locked("cpu")


def test_is_locked_stale_lock(tmp_path):
    """
    Test is_locked() with stale lock (process no longer exists).

    Objective: Verify is_locked() detects and ignores stale locks.
    Test criteria: Function returns False when lock file contains PID of non-existent process.
    How testing for this criteria fulfills purpose: Ensures dead process locks are cleaned up.
    What components are mocked, monkeypatched, or are fixtures: tmp_path fixture for isolated directory, mocked os.kill.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Simulates process death scenario.
    """
    locking.LOCKDIR = str(tmp_path)
    locking.LOCK_FPATH = {"cpu": os.path.join(locking.LOCKDIR, "cpu"), "gpu": os.path.join(locking.LOCKDIR, "gpu")}

    # Create lock file with non-existent PID
    lock_file = tmp_path / "cpu"
    lock_file.write_text("99999\n")

    with patch('os.kill', side_effect=OSError):  # Process doesn't exist
        assert not locking.is_locked("cpu")


def test_is_owned_owns_lock(tmp_path):
    """
    Test is_owned() when current process owns the lock.

    Objective: Verify is_owned() returns True when current process owns the lock.
    Test criteria: Function returns True when lock file contains current PID.
    How testing for this criteria fulfills purpose: Confirms ownership detection works.
    What components are mocked, monkeypatched, or are fixtures: tmp_path fixture for isolated directory.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Tests actual PID comparison.
    """
    locking.LOCKDIR = str(tmp_path)
    locking.LOCK_FPATH = {"cpu": os.path.join(locking.LOCKDIR, "cpu"), "gpu": os.path.join(locking.LOCKDIR, "gpu")}

    locking.lock("cpu")
    assert locking.is_owned("cpu")


def test_is_owned_does_not_own(tmp_path):
    """
    Test is_owned() when current process does not own the lock.

    Objective: Verify is_owned() returns False when another process owns the lock or no lock exists.
    Test criteria: Function returns False when lock file contains different PID or doesn't exist.
    How testing for this criteria fulfills purpose: Ensures proper ownership checking.
    What components are mocked, monkeypatched, or are fixtures: tmp_path fixture for isolated directory.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Tests actual file reading and PID comparison.
    """
    locking.LOCKDIR = str(tmp_path)
    locking.LOCK_FPATH = {"cpu": os.path.join(locking.LOCKDIR, "cpu"), "gpu": os.path.join(locking.LOCKDIR, "gpu")}

    # No lock exists
    assert not locking.is_owned("cpu")

    # Lock exists but owned by different process
    lock_file = tmp_path / "cpu"
    lock_file.write_text("99999\n")
    assert not locking.is_owned("cpu")


def test_unlock_not_owned(tmp_path):
    """
    Test unlock() when current process does not own the lock.

    Objective: Verify unlock() fails gracefully when process doesn't own the lock.
    Test criteria: Function returns False and doesn't remove lock file.
    How testing for this criteria fulfills purpose: Prevents unauthorized lock release.
    What components are mocked, monkeypatched, or are fixtures: tmp_path fixture for isolated directory.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Tests actual ownership check before unlock.
    """
    locking.LOCKDIR = str(tmp_path)
    locking.LOCK_FPATH = {"cpu": os.path.join(locking.LOCKDIR, "cpu"), "gpu": os.path.join(locking.LOCKDIR, "gpu")}

    # Lock owned by different process
    lock_file = tmp_path / "cpu"
    lock_file.write_text("99999\n")

    assert not locking.unlock("cpu")
    assert lock_file.exists()


def test_check_pause_no_pause_files(tmp_path):
    """
    Test check_pause() when no pause files exist.

    Objective: Verify check_pause() returns immediately when no pause files are present.
    Test criteria: Function returns without blocking when pause files don't exist.
    How testing for this criteria fulfills purpose: Ensures normal execution when no pause is requested.
    What components are mocked, monkeypatched, or are fixtures: tmp_path fixture for isolated directory, mocked time.sleep.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Simulates pause file absence.
    """
    locking.LOCKDIR = str(tmp_path)

    with patch('time.sleep') as mock_sleep:
        locking.check_pause()
        mock_sleep.assert_not_called()


def test_check_pause_process_specific(tmp_path):
    """
    Test check_pause() with process-specific pause file.

    Objective: Verify check_pause() blocks when process-specific pause file exists.
    Test criteria: Function calls sleep when pause file for current PID exists.
    How testing for this criteria fulfills purpose: Confirms pause mechanism works for specific processes.
    What components are mocked, monkeypatched, or are fixtures: tmp_path fixture for isolated directory, mocked time.sleep and os.getpid.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Simulates process-specific pause scenario.
    """
    locking.LOCKDIR = str(tmp_path)

    # Create process-specific pause file
    pause_file = tmp_path / f"pause_{os.getpid()}"
    pause_file.write_text("")

    with patch('time.sleep') as mock_sleep:
        # Mock to break after first sleep
        mock_sleep.side_effect = [None, KeyboardInterrupt]
        try:
            locking.check_pause()
        except KeyboardInterrupt:
            pass
        mock_sleep.assert_called()


def test_check_pause_global(tmp_path):
    """
    Test check_pause() with global pause file.

    Objective: Verify check_pause() blocks when global pause file exists.
    Test criteria: Function calls sleep when pause_all file exists.
    How testing for this criteria fulfills purpose: Confirms global pause mechanism works.
    What components are mocked, monkeypatched, or are fixtures: tmp_path fixture for isolated directory, mocked time.sleep.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Simulates global pause scenario.
    """
    locking.LOCKDIR = str(tmp_path)

    # Create global pause file
    pause_file = tmp_path / "pause_all"
    pause_file.write_text("")

    with patch('time.sleep') as mock_sleep:
        # Mock to break after first sleep
        mock_sleep.side_effect = [None, KeyboardInterrupt]
        try:
            locking.check_pause()
        except KeyboardInterrupt:
            pass
        mock_sleep.assert_called()


def test_lock_backoff_and_acquire(tmp_path):
    """
    Test lock() backoff behavior and eventual acquisition.

    Objective: Verify lock() waits and retries when resource is locked by another process.
    Test criteria: Function waits with backoff when lock is held by another process, then acquires when lock is released.
    How testing for this criteria fulfills purpose: Ensures proper concurrent access handling.
    What components are mocked, monkeypatched, or are fixtures: tmp_path fixture for isolated directory, mocked time.sleep.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Simulates lock contention scenario.
    """
    locking.LOCKDIR = str(tmp_path)
    locking.LOCK_FPATH = {"cpu": os.path.join(locking.LOCKDIR, "cpu"), "gpu": os.path.join(locking.LOCKDIR, "gpu")}

    # Create lock file simulating another process
    lock_file = tmp_path / "cpu"
    lock_file.write_text("99999\n")

    with patch('time.sleep') as mock_sleep, \
         patch('os.kill') as mock_kill:
        mock_kill.return_value = None  # Other process exists

        # Mock to simulate lock becoming available after a few tries
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:  # After 3 sleeps, remove the lock file
                lock_file.unlink()
            return None

        mock_sleep.side_effect = side_effect

        locking.lock("cpu")  # Should eventually acquire the lock
        assert locking.is_owned("cpu")
        assert mock_sleep.call_count >= 3