# -*- coding: utf-8 -*-
"""
Process-based resource locking mechanism for CPU and GPU resources.

This module provides a simple file-based locking system to coordinate access to 
shared resources (particularly CPU and GPU) across multiple processes. It uses 
files in a dedicated lock directory to track which process currently owns a 
particular resource lock.

Key features:
- Process-specific locks for both CPU and GPU resources
- Automatic backoff with exponential delay when resources are locked
- Lock ownership verification using process IDs
- Dead lock detection and cleanup if the owning process no longer exists
- Pause mechanism for temporarily halting specific processes

Typical usage:
    # Acquire lock before using a shared resource
    lock("gpu")
    
    try:
        # Use the resource...
        run_gpu_computation()
    finally:
        # Always release the lock when done
        unlock("gpu")

The locks are stored in the user's home directory under ~/locks/
"""

from pathlib import Path
import os
import random
import time
import unittest

# Lock directory in user's home directory
LOCKDIR = os.path.join(Path.home(), "locks")

# Lock file paths for different resources
LOCK_FPATH = {"cpu": os.path.join(LOCKDIR, "cpu"), "gpu": os.path.join(LOCKDIR, "gpu")}

# Initial backoff time in seconds when a lock is busy
BACKOFF_SECONDS = 1

# Create lock directory if it doesn't exist
os.makedirs(LOCKDIR, exist_ok=True)


def is_locked(device="gpu"):
    """
    Check if a resource is currently locked by another process.
    
    This function checks if the specified device is locked by another process.
    It reads the lock file to determine the process ID (PID) of the lock owner.
    If the current process owns the lock, it returns False (not locked from this
    process's perspective). If another process owns the lock, it verifies that
    the process still exists before considering the lock valid.
    
    Args:
        device: The resource to check ("gpu" or "cpu", defaults to "gpu")
        
    Returns:
        bool: True if the resource is locked by another active process,
              False if the resource is not locked or is locked by the current process,
              or if the locking process no longer exists (stale lock)
    """
    try:
        with open(LOCK_FPATH[device], "r") as f:
            # if (lock_owner := int(f.readline())) == os.getpid():  # not compat w/ python 3.6
            lock_owner = int(f.readline())
            if lock_owner == os.getpid():
                return False  # We own the lock, so it's not "locked" from our perspective
            try:
                os.kill(lock_owner, 0)  # Signal 0 doesn't kill the process, just checks if it exists
            except OSError:  # PID doesn't exist
                return False  # Stale lock (process no longer exists)
            return True  # Lock is owned by another active process
    except FileNotFoundError:
        return False  # No lock file exists


def is_owned(device="gpu"):
    """
    Check if the current process owns the lock for a specific device.
    
    This function verifies if the current process is the owner of the lock
    for the specified device by comparing the process ID in the lock file
    with the current process ID.
    
    Args:
        device: The resource to check ("gpu" or "cpu", defaults to "gpu")
        
    Returns:
        bool: True if the current process owns the lock,
              False if another process owns the lock or no lock exists
    """
    try:
        with open(LOCK_FPATH[device], "r") as f:
            if int(f.readline()) == os.getpid():
                return True  # Current process owns the lock
            return False  # Another process owns the lock
    except FileNotFoundError:
        return False  # No lock exists


def lock(device: str = "gpu"):
    """
    Acquire a lock for the specified device with exponential backoff.
    
    This function attempts to acquire a lock for the specified device. If the
    device is already locked by another process, it will wait using an exponential
    backoff strategy, gradually increasing the wait time between attempts.
    
    The function continues trying until it successfully acquires the lock. If it
    has been trying for a significant amount of time (approximately 9-11 backoff
    cycles), it will print status messages to inform the user of the wait.
    
    Args:
        device: The resource to lock ("gpu" or "cpu", defaults to "gpu")
        
    Returns:
        None: The function returns when the lock is successfully acquired
        
    Note:
        This function blocks until the lock is acquired, potentially indefinitely
        if the resource remains locked. Consider implementing a timeout mechanism
        for production use cases where indefinite blocking is undesirable.
    """
    # Start with initial backoff time
    backoff_time = BACKOFF_SECONDS
    
    # Keep trying until we own the lock
    while not is_owned(device):
        if is_locked(device):
            # Device is locked by another process, so wait
            time.sleep(int(backoff_time))
            
            # Increase backoff time (capped at 10 + random seconds or 10% increase, whichever is smaller)
            backoff_time = min(10 + random.random(), backoff_time * 1.1)
            
            # After about 9-11 iterations, print a status message
            if (
                backoff_time > BACKOFF_SECONDS * 1.1**9
                and backoff_time < BACKOFF_SECONDS * 1.1**11
            ):
                print("lock: spinning for %s..." % device)
        else:
            # Device is not locked, so we can acquire it
            with open(LOCK_FPATH[device], "w") as f:
                f.write(str(os.getpid()))
                
            # If we've been waiting a while, notify that we finally got the lock
            if backoff_time >= BACKOFF_SECONDS * 1.1**9:
                print("lock:ed %s." % device)


def unlock(device: str = "gpu"):
    """
    Release a lock for the specified device if owned by the current process.
    
    This function attempts to release a lock for the specified device, but only
    if the current process is the owner of the lock. This prevents a process
    from accidentally releasing another process's lock.
    
    Args:
        device: The resource to unlock ("gpu" or "cpu", defaults to "gpu")
        
    Returns:
        bool: True if the lock was successfully released,
              False if the lock was not owned by this process or didn't exist
              
    Note:
        It's good practice to always release locks in a finally block to ensure
        they are released even if an exception occurs during the locked operation.
    """
    if is_owned(device):
        # Remove the lock file if we own it
        os.remove(LOCK_FPATH[device])
        return True  # Successfully released the lock
    return False  # We don't own the lock, so we can't release it


def check_pause():
    """
    Pause execution when a process-specific or global pause file exists.
    
    This function allows controlled pausing of specific processes by creating
    pause files in the lock directory. It will pause execution (blocking) as
    long as either:
    - A process-specific pause file exists: ~/locks/pause_<PID>
    - A global pause file exists: ~/locks/pause_all
    
    The function will automatically continue execution when both pause files
    are removed. After 10 seconds of being paused, it will print a message
    indicating which file is causing the pause.
    
    Usage:
        To pause a specific process: Create a file named "pause_<PID>" in the LOCKDIR
        To pause all processes: Create a file named "pause_all" in the LOCKDIR
        
    Returns:
        None: The function returns when the pause is lifted
    """
    backoff_time = BACKOFF_SECONDS
    # Process-specific pause file based on current PID
    lock_fpath = os.path.join(LOCKDIR, "pause_%u" % os.getpid())
    # Universal pause file that affects all processes
    lock_fpath_universal = os.path.join(LOCKDIR, "pause_all")
    
    # Keep checking for pause files and wait if they exist
    while os.path.isfile(lock_fpath) or os.path.isfile(lock_fpath_universal):
        time.sleep(backoff_time)
        backoff_time += 1
        # After 10 seconds, notify the user that we're paused
        if backoff_time == BACKOFF_SECONDS + 10:
            print(f"paused by {lock_fpath} or {lock_fpath_universal}")


class Test_locking(unittest.TestCase):
    """
    Unit tests for the locking mechanism.
    
    This test class verifies the functionality of the process-based locking
    system, including lock acquisition, ownership checking, and lock release.
    It tests both the CPU and GPU lock types to ensure correct behavior.
    """
    
    def test_lock_unlock(self):
        """
        Test the lock acquisition, ownership verification, and release operations.
        
        This test verifies:
        1. Lock acquisition works correctly for both CPU and GPU
        2. Lock ownership is correctly reported
        3. Device-specific locks are independent (CPU vs GPU)
        4. Unlocking works when the process owns the lock
        5. Unlocking fails when the process doesn't own the lock
        6. is_locked() correctly identifies stale locks
        
        The test creates and releases locks, then manually modifies a lock file
        to test edge cases in lock validation.
        """
        # Acquire a CPU lock and verify ownership
        lock("cpu")
        self.assertTrue(is_owned("cpu"), "Process should own CPU lock after lock()")
        self.assertFalse(is_owned("gpu"), "Process should not own GPU lock yet")
        
        # Acquire a GPU lock and verify ownership
        lock("gpu")
        self.assertTrue(is_owned("gpu"), "Process should own GPU lock after lock()")
        self.assertFalse(is_locked("gpu"), "is_locked() should return False for owned locks")
        
        # Release GPU lock and verify result
        self.assertTrue(unlock("gpu"), "unlock() should return True when lock is released")
        
        # Simulate a stale lock by writing invalid PID to CPU lock file
        with open(LOCK_FPATH["cpu"], "w") as f:
            f.write("123")  # A likely non-existent PID
            
        # Verify behavior with invalid lock
        self.assertFalse(unlock("cpu"), "unlock() should fail when process doesn't own the lock")
        self.assertFalse(is_locked("cpu"), "is_locked() should detect and ignore stale locks")
        self.assertFalse(is_owned("cpu"), "Process should not own the lock after tampering")
        
        # Clean up
        os.remove(LOCK_FPATH["cpu"])


if __name__ == "__main__":
    unittest.main()
