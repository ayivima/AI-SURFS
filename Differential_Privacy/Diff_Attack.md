DIFFERENCING ATTACK: A SIMPLE OVERVIEW
======================================

At a very basic level, a DIFFERENCING ATTACK is when you use the DIFFERENCE between the results of two queries to find out information about an individual.

FOR EXAMPLE:
Consider the database of earnings of four people below. Assume, we wanted to find out how much Jane earns. But, we cannot have access to Jane's data directly. We can decide to find out how the sum of earnings change with and without Jane.

```
database = {"Jane":10000, "Doe":2000, "John":2500, "Dovy":3000}

sum1 = 0
for key in database:
    sum1 += database.get(key)
    
sum2 = 0
for key in database:
    if key!="Jane":
        sum2 += database.get(key)
        
print("Total Sum: ", sum1)
print("Sum without Jane: ", sum2)
print("Earnings of Jane: ", sum1-sum2)
```
OUTPUT:
```
Total Sum:  17500
Sum without Jane:  7500
Earnings of Jane:  10000
```
That simple...And, we just found out the earning of Jane without querying it directly: Rightly, 10000. This is how the DIFFERENCING ATTACK works.
If the data about people are different enough, when we take one person out, the query result can change and we can leverage this change to find out information about a targeted individual, without having direct access to the individual's data.

