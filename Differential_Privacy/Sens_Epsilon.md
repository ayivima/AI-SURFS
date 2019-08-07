

SENSITIVITY & EPSILON
=====================

SENSITIVITY
-----------
The maximum extent to which the output of a query to a database will change with the removal of a datapoint. 

Example:

- For, a database `A = [1, 6, 7, 19]`, the sensitivity of a sum query will become `19`
- For, a database `B = [-1, -2, -5, -10]`, the sensitivity of a sum query becomes `10`
- For, a database `C = [0, 1, 1, 0, 1, 1]`, the sensitivity of a sum query becomes `1`


EPSILON
-------

To simplify things, remember that we pick a random number from a range of numbers and add to the output of our query as noise, in differential privacy. 
Now, we just do not want any range. We want a range of numbers that is centered around a certain number, and we choose the Laplacian distribution for this purpose. 

Let's visualise.

We want to pick a random number to add to `3`, anytime someone asks for `3`. And, we want our random numbers to be centered around `0`.

We can decide on any of these range of numbers:

```
X = [-0.2, -0.1, 0, 0.1, 0.2]
Y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
Z = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

Notice that all these ranges of numbers center around `0`. 

In `X`, we have a narrow range of numbers. When we pick a number from `X` to add to `3`, the greatest changes will be `2.8` and `3.2` which are relatively close to `3`. The accuracy of the result will not be too affected. But, because the noised output will not be too far from the real 3, it will be easier to guess it...meaning less privacy.

`Z`, has the widest range of numbers. Using Z for your noise will mean `3` may be outputed as anything from `2.1` to `3.9`. Acuuracy reduces as `2.1` and `3.9` are relatively far from `3`


If you understand that, I used X, Y, Z to stand for the laplacian distribution...For easy understanding, just consider laplacian distribution as a range of numbers centered around a number for now. 

Then, Epsilon determines how narrow the distribution range is and, hence, how accurate or private the output will be.

EPSILON, PRIVACY, SENSITIVITY
-----------------------------

We can think of Epsilon in this light:

- High Epsilon - High accuracy, Low Privacy...because we are choosing noise from a narrower range of numbers.

- Low Epsilon - High Privacy, Low Accuracy...because we are choosing noise from a wider range of numbers.

Thus, in the light of privacy, the higher the sensitivity of the database query, the lower the epsilon should be and the vice versa. 

Bringing it to the technical definition, epsilon is what we use to control how much information leak we can permit. If it is high, it means we are trying to priotize accuracy, if it is low, we are trying to prioritize privacy. In between, we get a balance. It becomes more of a threshold for information leak.
