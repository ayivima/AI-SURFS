
Python is a general purpose, versatile, programming language first authored and released by Guido Van Rossum in 1991. 
It has risen tremendously in popularity, thanks to its inherent design's encouragement of simple, readable, 
elegant code. Today, python is useful for the small and big projects, the advanced and the very simple stuff, and to 
the beginner and pro programmer alike.

Remarkably, it has become a must-go-to for Data Science, Machine Learning, Deep learning, and Artificial Intelligence.
Let's explore some core python(CPython) uniqueness which is upheld by python lovers.



PYTHON UNIQUENESS HALL OF FAME
==============================

DO NOT DECLARE VARIABLE TYPES
----------------------------------------------------------

Unlike many classic C-style languages, python variables do not need declaration; they just need to be assigned. 
Thus, you do not need to specify variable types.
    
:Python 3:

::
    
    a_number = 1
    print(a_number)
    
    # OUTPUT: 1
    
:Java:
    
::
    
    public class PythonVrsJava{

     public static void main(String []args){
        int a_number = 1;
        System.out.println(a_number);
     }
     
    }
    
    // OUTPUT: 1


DO NOT USE CURLY BRACKETS - "{}" - FOR BLOCKS
---------------------------------------------

Unlike many classic C-style languages, python delineates blocks with just indentation(using spaces or tabs) and colon(":"),
instead of curly brackets.

:Python 3:

::
    
    a_number = 1
    
    if a_number == 1:
        print("Number is 1.")
    else:
        print("Number is not 1.")
    
    # OUTPUT: 
    # Number is 1.
    
:Java:
    
::

    public class PythonVrsJava{

      public static void main(String []args){
        int a_number = 1;
    
        if (a_number == 1){
            System.out.println("Number is 1.");
        } else {
            System.out.println("Number is not 1.");
        }
      }
     
     }
    
     // OUTPUT: Number is 1.


A VARIABLE CAN STORE DIFFERENT TYPES OF VALUES DURING ITS LIFETIME
-----------------------------------------------------------------

Unlike many classic C-style languages, python allows dynamic typing. Thus, you can assign a variable that previously stored an integer value, to a string value without throwing an error. Doing same in C will result in an error.

:Python 3:

::
    
    a_var = 1
    print(a_var)
       
    a_var = "Just a string"
    print(a_var)
    
    # OUTPUT: Just a string
    
:Java:
    
::
    
    public class PythonVrsJava{

     public static void main(String []args){
        int a_var = 1;
        System.out.println(a_var);
        
        a_var = "Just a string";
        System.out.println(a_var);
     }
     
    }
    
    // OUTPUT: 
    // PythonVrsJava.java:7: error: incompatible types: String cannot be converted to int
    //    a_var = "Just a string";
    

LOOPING THROUGH AN ITERABLE WITH FOR...IN RETURNS VALUES INSTEAD OF INDEXES
---------------------------------------------------------------------------

Unlike some classic C-style languages like Javascript which return indexes, Python returns values for ``for...in`` loops.

:Python 3:

::
    
    list1 = [1, 2, 3]
    
    for number in list1:
        print(number)
    
    
    # OUTPUT:
    # 1
    # 2
    # 3


:Javascript:

::

    let list1 = [1, 2, 3];
    
    for (let number in list1){
        console.log(number)
    }
    
    
    // OUTPUT:
    // 0
    // 1
    // 2
    
    

AN IMMUTABLE VALUE IS STORED IN ONLY ONE MEMORY LOCATION EVEN IF IT IS ASSIGNED TO SEPARATE VARIABLES
-----------------------------------------------------------------------------------------------------

:Python 3:

::
    
    num1 = 1
    num2 = 1
       
    str1 = "string"
    str2 = "string"
    
    bool1 = 3 == 2
    bool2 = "the" == "not"
    
    tuple1 = (1, 2, 3)
    tuple2 = (1, 2, 3)
    
    print("num1 address is {}".format(hex(id(num1))))
    print("num2 address is {}".format(hex(id(num2))))
    print("str1 address is {}".format(hex(id(str1))))
    print("str2 address is {}".format(hex(id(str2))))
    print("bool1 address is {}".format(hex(id(bool1))))
    print("bool2 address is {}".format(hex(id(bool2))))
    print("tuple1 address is {}".format(hex(id(tuple1))))
    print("tuple2 address is {}".format(hex(id(tuple2))))
    
    # OUTPUT:
    # num1 address is 0x5fefc880
    # num2 address is 0x5fefc880
    # str1 address is 0x3137b20
    # str2 address is 0x3137b20
    # bool1 address is 0x5fec71c0
    # bool2 address is 0x5fec71c0
    # tuple1 address is 0x32a26c0
    # tuple2 address is 0x32a26c0



UNLIKE IMMUTABLE VALUES, VALUES OF MUTABLE TYPES, LIKE LISTS AND DICTIONARIES, HAVE SEPARATE MEMORY ADDRESSES EVEN WHEN THEY ARE THE SAME FOR SEPARATE VARIABLES
------------------------------------------------------------------------------------------------------------------------------------

:Python 3:

::
    
    list1 = [1, 2, 3]
    list2 = [1, 2, 3]
    
    dict1 = {"a":1, "b":2}
    dict2 = {"a":1, "b":2}
    
    print("list1 address is {}".format(hex(id(list1))))
    print("list2 address is {}".format(hex(id(list2))))
    print("dict1 address is {}".format(hex(id(dict1))))
    print("dict2 address is {}".format(hex(id(dict2))))
    
    # OUTPUT: 
    # list1 address is 0xb445d0
    # list2 address is 0xb44a58
    # dict1 address is 0xb955d0
    # dict2 address is 0xb95630


A work in Progress...To be Continued


*Copyright 2019, Victor Mawusi Ayi. All Rights Reserved.*