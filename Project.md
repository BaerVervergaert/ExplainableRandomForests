# Features

- Data similarity based on the tree induced topology
  - Finished
  - Add custom aggregations over trees
  - Add custom aggregations over reference points
- Data extrapolation based on tree bounds (and training sample)
  - Connect to RF explainer
- Number of observations within tree bin
  - Connect to RF explainer
- Add parameter explanation for NGBoost learned parameters


# Notes

## Data similarity

Trees induce a topology on the input space. For each node (except leaves) we test if one of the features is lower or higher than a learned threshold value. Each node thus splits the space in half. The intersection of all splits of the visited nodes induce a bounding box (with possibly infinite bounds). All points in the bounding box receive the same output value from the tree. Thus, this bounding box is the neighborhood that is considered similar by the tree.

We can extrapolate this to a random forest with weights with the weighted average:
- Assume T is the set of trees
- Assume w_t is the weight of tree t
- Assume 1_bt (x) is the bounding box of tree t
- data similarity = sum(t in T) w_t 1_bt (x)

For multiple points we can apply another weighted mean. Other aggregations could also be interesting.

## Data extrapolation

Trees induce a topology on the input space. Each tree induces a bounding box with possibly infinite bounds. We can say a tree is extrapolating if the induced bounding box has one side with infinite bound. We could also include training data, and say a tree is extrapolating by choice for the area between the extreme of the data sample and areas where it is extrapolating. With training strategies leaving some features out we get the following flavours:
- A tree is extrapolating on x if the neighborhood of x has an infinite bound
- A tree is forced extrapolating on x if the neighborhood of x has only infinite bounds on features that were left out
- A tree is possibly forced extrapolating on x if the neighborhood of x has either two finite bounds or no bounds (infinite in both sides) on each feature
- A tree is extrapolating by choice on x if x is within the bounds of the training data extremes within the bounding box of x.

We can aggregate over the trees with a weighted mean to give us the extrapolating number.

Limitations:
- Extrapolation is potentially rare, so it has to be able to run for many points
- Because of the quantity speed is a concern
- We go through many trees so it shouldn't be too large
- Preferably it runs quickly

Integration plan:
- Because of the speed we might need to precalculate some values. This is acceptable, if not too memory intensive.
- Can we run through the decisions the tree makes?
  - We can see the nodes it has visited.
  - So, we only need to check the choice made at that point, or induce the choices.


