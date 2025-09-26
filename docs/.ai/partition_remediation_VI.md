- Have a look @/docs/.ai/partition_plan.md , especifically focus on lines 177-320. It is clear that this vision has not been completely realized.
- You will correct this architectural implementation quagmire.
- Much of the production logic _has_ been extracted and connected to the new clean interface - _some_ of it has _not_ and _some_ has been extracted, but not properly inserted into the new clean. There is no clean seperation: files may have completely inconsistent contents amongst these three possibilities.
- You are fixing a partial refactor, but do not assume anything like an  atomic update - this is most definitely NOT the case and would be a terrible assumption to make. You are going to need to be a lot more patient, and really go line by line on this if we stand any chance of figuring this out. Don't make assumptions about things being implemented because something else was implemented.

- The end result should  be consistent with the objective to minimize package interdependencies and will make use of clean interfaces for cross package communication - howver it should not sacrifice any funcitonality that would compromise the ability of the code to fulfill its inferred intent. (except for legacy CLI code - you can sacrifice that all you like)
- You should prefer to remove the legacy CLI whenever, rather than deprecate in place, whenever practicable in order to keep a clean codebase. Use PyCharm's advanced search tools to quickly determine whether something is safe to remove (or if it can easily be made safe to remove)
- You will use strict TDD, and ensure you understand the intent and structure of the codebase, package, class and function that you are working on prior to deciding how to edit it. This means that if you need to write failing tests, or modify existing tests resulting in failures to better match the spec in the partition_plan, then do that FIRST before writing your code / making your edits. Have a clear idea about what you want before you go about trying to get things
- You will then modify/extend/complete as necessary to use the clean interpackage interfaces, and above all - to realize the intent and vision of the codebase's author (me) as you understand it. 
- You are encouraged think laterally, propose alternate approaches, and consider changes elsewhere in the codebase if you believe it will better realize the author's intent and vision. Just be sure it is all down within the framework of red->green TDD.
- You shouldn't think of this as "wrapping" the extracted production logic
- You can rewrite, redistribute, change signatures, etc.
- The key insight is that the legacy code is like a handbook/cheatsheet for how things should be done
- The legacy code does NOT need to find its way into the final product verbatim
- This is a much more flexible and sensible approach. You should use the legacy code as a reference for understanding the domain logic, constraints, and proper implementations, but you are free to implement it in whatever way fits best with the clean API design.

---

# General rules
- Be sure to use the tools available to you as described in @/.roo/rules/rules.md
- Under no circumstances should you simplify or mock away real parts of the pipeline for integration testing. 
## Anti-patterns:
  - Never have more than one implementation of the same (or substantially similar) thing, 
  - Never call something by more than one name across the project unless it is absolutely necessary,
  - Never have more than one "sole source of truth" for any value.
  - The last thing to keep in mind is that you should streamline & cleanup (delete) legacy / duplicate code rather than leaving it deprecated in place

Here is some bonus guidance:

Do not try to immediately iterate test fixes. This is TDD; red -> green is normal. Instead think deeply about what we've observed from the test results and what larger incomplete-refactoring related issues those observations might be symptomatic of. Do not treat test failures as problems to be blindly resolved - remember your overarching big-picture goal and treat these test failures like invaluable pieces of data in determining what overall still needs to be done, and how to strategize and break that down into actionable chunks.

- You are correct in your strategic revelation that the tests are really architectural diagnostics, and I applaud you for that insight. Just because you have uncovered _some_ properly ported and functional production code does not mean you should abandon this revelation. Hold onto it dearly.

  ---

  Perfect! Now I can see the exact interface contract. Looking at the prepare_datasets method around line 652, I can see:

Training expects: trainer.prepare_datasets(dataset_config) to return Dict[str, Iterator] with keys like 'train_loader', 'val_loader', 'test_loader'

Training calls: create_training_datasets() from the dataset package, and expects it to return a dict with:

'train_dataloader'
'validation_dataloader'
'test_dataloader'
Interface mapping: The training code maps these to:

'train_loader': dataset_info['train_dataloader']
'val_loader': dataset_info['validation_dataloader']
'test_loader': dataset_info['test_dataloader']
Now let me see what create_training_datasets in the dataset package actually provides by looking at the end of that function that I saw earlier. The key insight is that this should work, but there might be:

Multiple implementations (anti-pattern violation)
Missing functionality in the dataset package
Interface mismatches between what training expects vs what dataset provides
Let me methodically examine the current dataset package create_training_datasets function to see if it actually provides the expected interface.

But more importantly, the user is emphasizing the strategic insight that I need to:

Understand the refactoring scope from partition_plan.md lines 243-267
Use legacy as handbook for domain knowledge, not code to migrate verbatim
Delete duplicates rather than deprecate in place
Focus on anti-patterns as root cause diagnostics
Let me first understand what SHOULD be in the dataset package according to partition_plan.md lines 243-267, then check what's actually there.

I notice critical class distribution issues across multiple dataset files. There are clear signs of code duplication with classes like ProfiledRGBBayerImageDataset appearing in multiple locations. This suggests a need for consolidation and careful architectural review of the dataset package structure.

The key observation is identifying where classes are redundantly implemented and determining the most appropriate single location for each core dataset class. Consolidation will help reduce maintenance overhead and potential inconsistencies in dataset handling. I see the core problem with our codebase's structure. We need a systematic cleanup to resolve class duplication and improve code organization. The key actions involve consolidating implementations, removing redundant files, and ensuring clean, consistent interfaces across our dataset and training packages. This will streamline our code and prevent potential import and implementation conflicts.

Our strategic cleanup involves precise file management, methodical class extraction, and eliminating legacy code that introduces unnecessary complexity. By focusing on a single, authoritative source for each class, we'll create a more maintainable and predictable system.


**PERFECT INTERFACE DISCOVERY!** ✅ Now I can see the **exact contract mismatch**:

**Training [`prepare_datasets()`](src/rawnind/training/clean_api.py:676) expects**:
- Calls [`create_training_datasets()`](src/rawnind/dataset/clean_api.py:777) from dataset package
- Maps: `'train_dataloader'` → `'train_loader'`, `'validation_dataloader'` → `'val_loader'`

**But the fundamental issue** is **ANTI-PATTERN VIOLATIONS**:

✅ **Extracted to proper files**: [`RawImageDataset`](src/rawnind/dataset/base_dataset.py:47) → [`base_dataset.py`](src/rawnind/dataset/base_dataset.py)
❌ **DUPLICATE**: [`ProfiledRGBBayerImageDataset`](src/rawnind/dataset/bayer_datasets.py:19) in **TWO files** ([`base_dataset.py`](src/rawnind/dataset/base_dataset.py:248) + [`bayer_datasets.py`](src/rawnind/dataset/bayer_datasets.py:19))
❌ **TRIPLE DUPLICATE**: [`TestDataLoader`](src/rawnind/dataset/test_dataloaders.py:27) in **THREE files** ([`base_dataset.py`](src/rawnind/dataset/base_dataset.py:339) + [`test_dataloaders.py`](src/rawnind/dataset/test_dataloaders.py:27) + [`legacy_rawds.py`](legacy_rawds.py:270))

Let me **systematically eliminate duplicates** starting with the **most critical one**. Which file should be the **single source of truth** for [`TestDataLoader`](src/rawnind/dataset/test_dataloaders.py:27)?


---

---

The user wants me to:

Finish dealing with the anti-patterns I've discovered
Then perform the whole discovery process again on what's left to make sure we don't miss any
Great approach! So I need to:

Continue eliminating the duplicates I found:
ProfiledRGBBayerImageDataset (in base_dataset.py line 248, keep bayer_datasets.py version)
CleanCleanImageDataset (move from base_dataset.py line 280 to clean_datasets.py)
CleanNoisyDataset (move from base_dataset.py line 319 to noisy_datasets.py)
Then do a comprehensive pass over all modules to find any other anti-patterns