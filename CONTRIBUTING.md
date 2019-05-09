# Contributing


This document explains how we make a change to the code base.

## General information

The code in the master branch is meant to be stable. Separate features should be developed in feature-branches.
There is a magic for feature branches: if a branch starts with `experiment` it will be automatically evaluated by
[evaluator](https://github.com/synthesized-io/evaluator) (See the README.md to get information about evaluation 
results).

Quick fixes and changes outside of `synthesized.core` can be pushed straight to the master if there is an urgent need.
If you do so, please notify colleagues and provide a reason for that.

## Working on a new feature

* Create a branch for the feature.
* If you want results be evaluated add `experiment` to the name.
* Push logical steps of the tasks as separate commits with meaningful messages.
* Work considered done when you found evidence that change has a positive effect.
Check evaluation, perform more detailed analysis if needed.
* Avoid long-living branches. Try to split your work into milestones and integrate your changes into master rather sooner.
* When work is done create a Pull Request and ask somebody to review it.
* If a pull request is created, the feature is considered finished and other people can review it.
* If there is a need to get early feedback for a change before work is done, it's fine to create
a pull request, but you should add `[WIP]` (work in progress) to the name of that PR.
* If the review is done you can merge changes to the master. Note that changes are typically merged by the creator of the PR.

## Memo for a reviewer

* Understand the purpose of the change.
* Check if the code passes tests on the CI.
* If the branch is an experiment branch then check results of the evaluation
* If tests failed there is no need for a review - code should be fixed first.
* Read out the code and leave comments.
* Approve the PR (there is a button in the GitHub interface)

## Python coding guidelines
* Use type annotations. You should annotate at least method signatures.
* Avoid comments in docstrings if they can be replaced with meaningful method/variable names and type annotations.
* If you write a docstring it should rather explain the purpose of the code and design considerations. I.e. it should provide a context for the reader and should not duplicate the code itself.
* Note about names arguments: make use of them by default but avoid when the meaning of an argument is obvious. For example it's better to write `tf.sin(x)` than `tf.sin(x=x)` but in more complex cases prefer named arguments. There is no strict rule when argument name is obvious therefore use your best judgment.
* We are supporting python >= 3.6 (3.5 has a weak type annotations support)
