# Introduction

This package provides an explainability tool to use in analysis of random forest models.

The core functions provided are:
- a data similarity score
- a data extrapolation score
- a data quantity score

# Explanation

All scores are based on the topology induced by the underlying decision trees of the random forest models.

The data similarity score informs on how similar a data sample is to a reference point (or several reference points) based on how often these end in the same leaf of the decision tree.

The data extrapolation score informs on how much a data sample is extrapolating by tracking the distance to the decision bounds it encountered.

The data

# Examples



# Installation

Package not yet available on PyPi
