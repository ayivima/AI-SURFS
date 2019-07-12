THE SOFTMAX FUNCTION
====================

Primarily, the softmax function is a mathematical function which has been borrowed for use as activation functions in artificial neurons.
So the principles of the softmax function remains mathematical.


WHAT DOES SOFTMAX DO?
---------------------
At a basic level, the softmax function takes a sequence of numbers, or an array of numbers, and operates on them such that now all of them add up to 1.

**For example:**
> for `X = [a, b, c]`
> the sum of X, `a + b + c`, can be any number.

> With softmax function applied to X, we get `Xsoft = [m, n, o]`, such that the sum `m + n + o` equals 1.
> So in simple terms, when we want to change a sequence of values so that they all fall within the range 0 and 1, and their sum equals 1, we can use the softmax function.

HOW DOES IT WORK
----------------
Given `X = [a, b, c]`,

1. Apply the exponential function to every number in the sequence:
`[exp(a), exp(b), exp(c)]`

2. Get the sum, SUM_OF_EXPS, of the "exponentials":
 `exp(a) + exp(b) + exp(c)`

3. Divide every "exponential"  by the SUM of exponentials obtained in step 2:
`[exp(a)/SUM_OF_EXPS, exp(b)/SUM_OF_EXPS, exp(c)/SUM_OF_EXPS]`

**Bingo!!!**


HOW IS THIS USEFUL IN NEURAL NETWORKS?
--------------------------------------
When a neural network has sifted through data, got the weighted sums, we need an output that "says" something meaningful. We do not just need the numbers, as they might not mean anything.
In NNs, when we want to classify things, we must draw inferences from the probabilities of the given entity belonging to each of our target classes. This is especially useful, when the entity can belong to more than two classes.

**For example:**
> If we have 3 classes of colours: `[red, green, blue]`, and our neural network has 3 outputs each for a given colour.

> For a particular colour which is to be determined, the outputs are as follows `[0.758, -0.875, 0.654]` for red, green, blue respectively. Which of these colours will we classify our unknown colour?
> We cannot tell yet, as some values are negative whilst others are positive. Thus, we need to find a level ground to compare them. We decide that we want to see where each of them will fall within the range 0 to 1.
> Then, we can have a fair comparison. Softmax becomes useful, because it will change all these numbers into positive numbers falling within the range of 0 and 1, then we can take the highest number as the highest probability.

> When we apply softmax to our output, `[0.758, -0.875, 0.654]`, we get `[0.477, 0.093, 0.430]`, for RED, GREEN, BLUE respectively. Clearly, we can conclude that `RED` has the highest probability.
> With this scenario, the NN can classify the colour as `RED`.


ROUND UP
--------
The softmax function makes it easy for an NN to get a class for an entity by transforming final outputs into a range of values between `0` and `1`.
These transformed values become probabilities for the respective classes, and the class with highest probability is assigned. It is especially useful when an entity can belong to several classes.
