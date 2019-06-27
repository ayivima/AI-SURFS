TENSORS: ARE THEY SCALARS, VECTORS, MATRICES?
===============================================

In past times, tensors would probably be most popular among faithfuls of the study of the human body, and especially of the musculoskeletal system. As modern technology keeps at breaking limits through artificial intelligence, however, this term is becoming mainstream, and the trend frequently stirs up the question: What is a tensor?

What is a tensor?
=================

Given the complexity with defining a tensor, I will approach it in steps.

Step 1
------
Firstly, appreciate a tensor as the mathematical representation of an interest.
An interest can be a color of an object, speed of a car, price of an ice cream, nationality of a person, shape of someone's nose.

Certainly, Yes! Everything can be expressed in numbers. But how?
Supposing we want to categorise 4 types of noses:
* Fleshy Nose
* Bumpy Nose
* Snub Nose
* Hawk Nose

Fleshy Nose, being the first type, can be represented by `1`, Bumpy Nose `2`, Snub Nose `3`, Hawk Nose `4`. Afterwards, if I was sent the number `1`, and told it was a type of nose, I would automatically know it was referring to a Fleshy Nose.

Why do we have to represent things in numbers anyway? 

Firstly, numbers are language agnostic. In otherwords, irrespective of whether a person only understands greek, latin, french, english or swahili among others, we all understand numbers. Thus, it becomes easier to pass information across varied backgrounds using numbers, as it is a sort of universal language.

Secondly, computers understand only zeros and ones (numbers) and at the end of it all, whatever we want to do with our computers will get converted to numbers. So numbers are foundational in computing.

Back to the noses example,
`1` stands for Fleshy nose. Thus, `1` is a tensor representing a type of nose. Furthermore, `1` is a type of tensor called scalar, because it has just one dimension - it is just one number. Step two talks about another type of tensor that can have more dimensions.

Step 2
------
Appreciate that interests are multifaceted in real life. In otherwords, interests are made of several parts or dimensions. 

Let's take the price of an ice cream. Assume it is $5.00.
You might think it has just one dimension, but it has several dimensions:
* the currency sign ($)
* the first group of numbers before the dot (5)
* the dot (it could have been a comma in another locale)
* the group of numbers after the dot

We can rewrite the price in a group of numbers.
* Let's assign a number to currency. There are about 180 currencies in the world and we can decide to assign `1` as the US dollar currency. Subsequently, anytime we see `1` representing a currency, it's a US dollar.
* We can decide to leave the group of numbers before the dot or comma as is, since they are already numbers
* We can assume that the position of the dot can be occupied by just two symbols (technically, decimal separators): dot and comma. Thus, dot becomes `1`, and comma `2`.
* We will leave the numbers after the dot as is
* Finally, lets use the format, ``(currency, numbers before dot/comma, dot/comma, numbers after dot/comma)`` to rewrite the price of the ice cream, in its four dimensions, like this:
```
price_of_ice_cream = (1, 5, 1, 0)
```
In a locale, like Canada(French), where a comma is used as a decimal separator, if ice cream cost 5 dollars, this could be written as $5,00. Let's assign 2 to Canadian dollar. The price could be written as:
```
price_of_ice_cream = (2, 5, 2, 0)
```
The first `2` is for the canadian dollar, the second value `5` for the first part of price value, the third value `2` for the comma (decimal separator), and `0` for the last part of the price value.

What we just did was to represent the price of an ice cream as a tensor. This time it is a vector, not a scalar, and it is a four-dimensional vector because it uses four numbers (dimensions). Depending on what we want to do, it could become a two dimensional vector (x, y), three dimensional (x, y, z), or even seven dimensional (x, y, z, a, b, c, d), if we gleaned more dimensions from the price.

Step 3
------
Appreciate that an object or event can have multiple points of interests, which themselves have several dimensions.

In describing an ice cream, points of interest can include:
* price
* producer
* flavor

Let's assume the following for one ice cream(Ice Cream A):
* Price = US$5.00
* Producer is Dovry Ice Creams, with a producer ID of 0052
* The flavor is strawberry - the strawberry flavor has a code of 2153

Let's assume these for another ice cream(Ice Cream B):
* Price = CAN$5,00
* Producer is Resty Ice Creams, with a producer ID of 1045
* The flavor is strawberry, maitaining the code of 2153

We already have the vectors for price. We can easily create vectors for producer from the digits of the ID, and flavor from the digits of flavor code.

Ice cream A description in vectors:
```
# price is given in step 2 above
price_of_ice_cream_A = (1, 5, 1, 0)
producer_of_ice_cream_A = (0, 0, 5, 2)
flavor_of_ice_cream_A = (2, 1, 5, 3)

```
Ice cream B description in vectors
```
# price is given in step 2 above
price_of_ice_cream_A = (2, 5, 2, 0)
producer_of_ice_cream_A = (1, 0, 4, 5)
flavor_of_ice_cream_A = (2, 1, 5, 3)

```
But we need to to be able to transport all vectors related to a particular ice cream together.

We might end up with these:
```
ice_cream_A = [(1, 5, 1, 0), (0, 0, 5, 2), (2, 1, 5, 3)]
ice_cream_B = [(2, 5, 2, 0), (1, 0, 4, 5), (2, 1, 5, 3)]
```
And we can rewrite this more beautifully:
```
ice_cream_A = [(1, 5, 1, 0), 
               (0, 0, 5, 2), 
               (2, 1, 5, 3)]
               
ice_cream_B = [(2, 5, 2, 0),
               (1, 0, 4, 5),
               (2, 1, 5, 3)]
               
```
These are also tensors. This time they are tensors carrying information from several vectors, and they look like tables. Each vector for a particular ice cream becomes a row in a tensor, and the dimensions form columns. This type of tensor is called a matrix(plural: matrices).

We can remove the brackets and commas to clarify the matrices:
```
ice_cream_A = [ 1 5 1 0 
                0 0 5 2 
                2 1 5 3 ]
               
ice_cream_B = [ 2 5 2 0
                1 0 4 5
                2 1 5 3 ]
               
```
There we go! Tensors can be matrices formed from several vectors. If it helps, you can look at matrices like tables of information.

Step 4
------
Appreciate that often times we want to carry information about more than one thing around.

In step 3, we used matrix to carry several information about only one particular ice cream. However, what is frequently done is to use matrices to represent information about the same interest from different objects. For example, if we are interested in comparing prices, a matrix can carry information about prices of different things.

Let's illustrate using the two ice cream prices in step 2:

We can have a matrix called prices_of_ice_creams and it will have the two prices US$5.00, CAN$5,00 as demonstrated below:
```
prices_of_ice_creams = [(1, 5, 1, 0), 
                        (2, 5, 2, 0)]
                        
```
Let's clean it up:
```
prices_of_ice_creams = [ 1 5 1 0 
                         2 5 2 0 ]
                        
```
And we have another matrix with 2 vectors(rows) having four dimensions(columns). This is a tensor, a 2 X 4 matrix, as it has 2 rows and 4 columns.

Takeaways
=========
* Tensors represent things or objects mathematically
* If the number representing an object is single, in otherwords, it has one dimension, it is called a `scalar`. For example:
   ```
   dollar_sign = 1
   ```
* If a particular representation has more than one dimension, this tensor is called a vector. For example:
   ```
   price_of_ice_cream_A = (1, 5, 1, 0)
   ```
* When a tensor has information about several things, like prices of several ice creams, it is called a matrix. A matrix can be seen as a table, with each row carrying information about a particular item. For example:
    ```
    prices_of_ice_creams = [ 1 5 1 0 
                             2 5 2 0 ]
    ```
    Here each row contains the price information of one ice cream.
* Tensors can be scalars, vectors, matrices or a combination of them.
* You can look at vectors as a group of scalars.
* You can look at matrices as a group of vectors.
* A vector is a sequence of numbers; an array, list or tuple etc, depending on the programming language used. The number of dimensions of a vector refers to the number of items in it. `[1, 2, 3]` is a 3-dimensional vector, and `[2, 3, 4, 5, 6]` is a 5-dimensional vector.
* A matrix is like a table and can have rows and columns.
* A matrix can be described by the number of rows and columns. The following has `2` rows and `2` columns, and is called a 2X2 matrix. The first row is `1, 5` and the second row is `2, 3`. The first column, from up to down, has `1, 2` and the second column has `5, 3`.
```
a_matrix = [ 1 5
             2 3 ]
```

Given this mathematical introduction to tensors, we can further explore its adoption in computer science. For example, in Tensorflow, we might come across a tensor that contains text and not numbers...Oops! Fret not! It all comes down to numbers along the line anyway, and an understanding of the numerical foundations of computing, it will all fall in place.

Happy AI programming!

