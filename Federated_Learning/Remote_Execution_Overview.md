


AN OVERVIEW OF REMOTE EXECUTION IN PYSYFT
=========================================

An attempt at explaining, in simple terms, Remote Execution with PySyft. It becomes quite a simple process if we break it down in steps.


What is remote execution?
-------------------------
Remote computing allows us to perform tasks on another person's computer on our network. More simply, if I wanted to do a computation, and I cannot run it on my computer, I can do it remotely on another computer. A close example is how we train our models in the Udacity workspace or Google colab. We can see how mainstream this has become.

When it comes to federated learning, we want to use training outcomes from several devices to make predictions. The only way we can do this is to be able to train and send outcomes remotely. 
This is what is demonstrated by the Bob example. 


What are Pointers?
------------------

When we store something in a computer's memory, how do we know where it is?

That's the work of pointers. Pointers store the addresses, or locations, of the things we store in memory. When it is time to get back a stored value, a pointer for the value we look for is checked for the address, and then the value can be fetched.

In federated learning, this is particularly useful because our resources, tensors and model resources etc, are stored on different devices. This means we need to locally know where what is, then we can get them when needed. Thus, pointers reference our various connections, or other devices we communicate to, if i should put it, so that our models can correspond with them when needed.


What are Virtual Workers?
-------------------------

Virtual workers are the heart of federated learning. They are responsible for running all the "errands". In otherwords, they are the objects that do the real communication between devices, sending commands and receiving outcomes.


What's the link between all these?
----------------------------------

Deep learning is the trend today. But, privacy issues continue to linger. Thus, we need to go at all odds to ensure that we do not cause harm to people by using their data to train our models. One of the emerging ways to achieve this is federated learning. Through federated learning, we give users control over their data. Instead of asking them to send us their data for our models to use in training and making predictions or classification, we ask them to keep their data, perform the training and other tasks on their devices, and send us the outcomes. This way we do not get to invade their privacy. Thus, we execute the tasks remotely on their devices, and everybody gets happy.
