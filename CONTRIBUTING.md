# Contributing


This document explains how we make a change to the code base.

## General information

The code in the master branch is meant to be stable. Separate features should be developed in feature-branches.
There is a magic for feature branches: if a branch starts with `experiment` it will be automatically evaluated by
[evaluator](https://github.com/synthesized-io/evaluator) (See the README.md to get information about evaluation 
results).

There is a special `dev` branch which is used to develop the next version of architecture consisted of many features
that can not be done in isolation.

Quick fixes and changes outside of `synthesized.core` can be pushed straight to the master if there is an urgent need.
If you do so, please notify colleagues and provide a reason for that.

## Working on a new feature

* Create a branch for the feature.
* If you want results be evaluated add `experiment` to the name.
* Push logical steps of the tasks as separate commits with meaningful messages.
* Work considered done when you found evidence that change has a positive effect.
Check evaluation, perform more detailed analysis if needed.
* When work is done create a Pull Request and ask somebody to review it.
* If pull request is created, the feature is considered finished and other people can review it.
* If there is a need to get early feedback for a change before work is done it's fine to create
a pull request but you should add `[WIP]` (work in progress) to the name of that PR.
* If the review is done you can merge changes to the master. Note that changes are typically merged by the creator of the PR.

## Memo for a reviewer

* Understand the purpose of the change.
* Check if the code passes tests on the CI.
* If the branch is an experiment branch then check results of the evaluation
* If tests failed there is no need for a review - code should be fixed first.
* Read out the code and leave comments.
* Approve the PR (there is a button in the GitHub interface)

## Policy of working with dev branch

* It's for "big experiments" which consist of many changes.
* It can hang for a quite long time.
* We merge it if we have a joint agreement.
