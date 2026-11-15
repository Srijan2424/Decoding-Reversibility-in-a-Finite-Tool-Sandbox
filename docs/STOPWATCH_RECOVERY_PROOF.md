# Stopwatch activation recovery certificate

This certificate concerns the pinned ScienceWorld revision e8216d6044e8e39be9fcb185e3b2dfb602584b52, boil variation 2. The recorded before-state has stopwatch object 17298 with current and displayed counters of 0. The legal action `activate stopwatch` changes them to 1. A direct `deactivate stopwatch` resets the current counter to 0 and leaves the displayed counter at 1. See `pilot.json`.

## Why no sequence restores the recorded zero display

- The only assignment to the display after object construction copies the current counter after incrementing it in an activated tick. At the project step limit this is strictly positive. The deactivated tick clears the internal current counter and preserves the display. An audit of the pinned Scala sources found no other assignment to either stopwatch counter outside this class.
- An object created later receives a different UUID from the monotonically increasing allocator. Removing or replacing the stopwatch therefore cannot restore the same original object and reading. The allocator resets in the simulator load/reset path, which is a replay operation. The available in-game `reset task` command only resets goal progress and focus.
- Therefore any legal recovery through three further calls fails at least one necessary condition for exact recovery: the same object must still exist and display zero. This rules out exact recovery without requiring enumeration. If the partial-recovery policy allows only temperature differences, the display discrepancy also rules out partial recovery under that policy.

This is one source-backed case with a reproducible candidate transition. It has no final dataset label because the study has not validated a shared partial-recovery rule for ScienceWorld. It does not resolve the freeze door case.

Source files at the pinned revision:

- [StopWatch.scala](https://github.com/allenai/ScienceWorld/blob/e8216d6044e8e39be9fcb185e3b2dfb602584b52/simulator/src/main/scala/scienceworld/objects/devices/StopWatch.scala)
- [UniqueIdentifier.scala](https://github.com/allenai/ScienceWorld/blob/e8216d6044e8e39be9fcb185e3b2dfb602584b52/simulator/src/main/scala/util/UniqueIdentifier.scala)
- [ActionFocus.scala](https://github.com/allenai/ScienceWorld/blob/e8216d6044e8e39be9fcb185e3b2dfb602584b52/simulator/src/main/scala/scienceworld/actions/ActionFocus.scala)
