# Python Decorators For Data Engineers and Data Scientists

Enhance the Readability, Efficiency, and Code Management of your Python skill by incorporating these wrappers for a significant upgrade.

I'm a data engineer and machine learning engineer by train, so no doubt that Python is by far my favourite programming language due to its simple syntax and powerful application in various domains. I have been working with Python and plenty of Python libraries for Pyspark all the time in the last 8 years as the job required. Python is easy to learn and because it is too easy to learn, a lot of people just jump right into it and start to write a function that does its job without considering how to use Python in a good way that actually saves time and effort.

![Photo by [Chris Ried](https://unsplash.com/@cdr6934?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://miro.medium.com/0*0UoIabQBbZ4jb7jT)

I remember when I switch to Python from Java and C++, it was so beautiful, and back in the day, working with data with Pandas is dead easy. You just need to write a function that does its job and is done. But then when I get into bigger and bigger projects, I realized that I've used Python in not a good way. Without class and OOP, my project keep stacking higher and higher all the similar functions that do pretty much the same job.

Those days along with the old Python version are long gone. Every project I started either a data processing project or PySpark project for big data pipeline or just a Web API project I spun up for fun, I always make sure to follow best practices and leverage the full potential of Python's capabilities. The transition from writing simple functions to adopting object-oriented programming (OOP) was a turning point in my Python journey.

In addition to adopting OOP, I also recognized the importance of writing clean, readable, and well-documented code. Commenting on complex logic, using meaningful variable and function names, and adhering to style guidelines have become second nature to me.

![](https://miro.medium.com/0*sdtOL4uScBVwme5n.jpg)

As the Python ecosystem continued to evolve, I found myself exploring newer libraries and frameworks that further streamlined my work, especially the _**decorator**_ - a design pattern that allows you to modify the functionality of a function by wrapping it in another function.

Funny that decorators were seldom on my radar unless absolutely necessary, like employing them for handling authentication and authorization in web APIs. However, as I delved deeper into the concept, I realized how much they could enhance code organization and reusability. In the realm of data engineering, where the focus lies on building robust and efficient data pipelines, decorators found their unique niche. They became a game-changer when it came to data validation and quality control. By creating decorators that automatically check data integrity before it enters the pipeline, I drastically improved the reliability of the entire process.

Therefore, in this short article, we'll explore the concept of Python wrappers and present examples that can improve our Python development process.

## Python Wrappers

Python wrappers refer to functions integrated into another function, enabling the addition of extra functionality or alteration of behaviour without directly editing the original source code. Typically implemented as decorators, these special functions accept another function as input and enact changes to its operation.

Wrapper functions find utility in various contexts:

- **Extending Functionality**: Through the application of decorators, functionalities such as logging, performance measurement, or caching can be seamlessly incorporated into functions.

- **Promoting Code Reusability**: Wrapper functions, and even classes, can be universally applied to multiple components, eliminating redundancy and guaranteeing consistent behaviour across diverse elements.

- **Modifying Behavior**: Wrapper functions allow for the interception of input arguments. For instance, input validation can be conducted without the need for numerous assert statements, thus enhancing code clarity.

## Steps to create a decorator

### Define the Decorator Function

Start by defining a function that will act as your decorator. This function should take another function (the function to be decorated) as its argument. Typically, decorators have an inner function that performs the actual modification.

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        # Code to execute before the decorated function
        result = func(*args, **kwargs)
        # Code to execute after the decorated function
        return result
    return wrapper
```

1. **Use the Decorator Inside the Wrapper Function**: 
Within the decorator's inner function (`wrapper` in the example above), you can perform actions before and after the decorated function is called. This could involve logging, performance measurement, or any other desired behaviour.

2. **Invoke the Decorated Function Inside the Wrapper**: 
Call the original function (`func` in the example above) from within the wrapper. You can pass any arguments and keyword arguments that were originally meant for the function.

3. **Return the Wrapper Function**: 
The decorator function should return the inner `wrapper` function. This is crucial because the returned function will replace the original function when the decorator is applied.

4. **Apply the Decorator Using the "@" Symbol**: 
To use the decorator, apply it to the function you want to modify using the "@" symbol. Place the decorator's name above the function definition.

```python
@my_decorator
def my_function():
    # Function code
```

## Show Time

The better way to learn is to practice. So let's go through some of the examples that simplify real-world scenarios.

### 1. Timer

Easy and straightforward, I always have this custom decorator for every project. This decorator is used to monitor how long it takes for a particular function to complete data processing.

```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        # start the time measure
        start_time = time.time()
        # call the function
        result = func(*args, **kwargs)
        # end of processing
        end_time = time.time()
        # calculate the difference
        exec_time = end_time - start_time

        print(f"===== INFO: Execution time: {exec_time} seconds")
        # return the result of the decorated function
        return result
    # return reference to the wrapper function
    return wrapper
  
```

Creating a decorator in Python involves defining a function named **`timer`** which accepts the parameter **`func`** to indicate that it is a decorator function. Inside the **`timer`**function, establish another function named **`wrapper`** which receives the customary arguments passed to the targeted function intended for decoration.

Inside the "wrapper" function, the desired function is invoked using the supplied arguments **`result = func(*args, **kwargs)`**.

The wrapper function returns the result of the decorated function's execution. The **decorator** function should return a **reference** to the **wrapper** function we just created.

Now, just need to apply it to the desired function using the `@` symbol.

```python
@timer
def data_processing():
    print("Starting the data processing..")
    # simulate a function by pausing for 5 seconds
    time.sleep(5) 
    print("Finished processing")

data_processing() 
```

**Output:**

> Starting the data processing..
Finished processing
===== INFO: Execution time: 5.006768703460693 seconds

Apart from measuring data processing time, I also use this function to log the processing time of the API endpoint.

### 2. Retry with Parameters

The provided wrapper retries the execution of a function for a designated number of attempts, incorporating pauses between each retry. This mechanism proves valuable when handling network or API calls prone to sporadic failures caused by temporary issues.

You see that in the example below, I pass the arguments to the decorator itself. In order to achieve this, we define a decorator maker that accepts arguments and then define a decorator inside it. We then define a wrapper function inside the decorator as we did earlier.

```python
import time

def retry(max_attempts, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed: {e}")
                    time.sleep(delay)
            print(f"Function failed after {max_attempts} attempts")
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def call_endpoint(url):
    print("Call the API endpoint to get the data..")
    # raise error to simulate that server is not responding
    raise TimeoutError("Server is not responding")

call_endpoint("https://anyapiendpoint.com/data/34234234")
```

> Output:
Call the API endpoint to get the data..
Attempt 1 failed: Server is not responding
Call the API endpoint to get the data..
Attempt 2 failed: Server is not responding
Call the API endpoint to get the data..
Attempt 3 failed: Server is not responding
Function failed after 3 attempts

### 3. Trace Stack

The decorator **`@stacktrace`** is used to emit useful messages when functions are called and values are returned from functions:

```python

from sys import settrace

def stacktrace(func=None, exclude_files=['2023-08-22']):
    def tracer_func(frame, event, arg):
        co = frame.f_code
        func_name = co.co_name
        caller_filename = frame.f_back.f_code.co_filename
        if func_name == 'write':
            return # ignore write() calls from print statements
        for file in exclude_files:
            if file in caller_filename:
                return # ignore in ipython notebooks
        args = str(tuple([frame.f_locals[arg] for arg in frame.f_code.co_varnames]))
        if args.endswith(',)'):
            args = args[:-2] + ')'
        if event == 'call':
            print(f'--> Executing: {func_name}{args}')
            return tracer_func
        elif event == 'return':
            print(f'--> Returning: {func_name}{args} -> {repr(arg)}')
        return
    
    def decorator(func: callable):
        def inner(*args, **kwargs):
            settrace(tracer_func)
            func(*args, **kwargs)
            settrace(None)
        return inner
    if func is None:
        # decorator was used like @stacktrace(...)
        return decorator
    else:
        # decorator was used like @stacktrace, without parens
        return decorator(func)
```

With this, we can put on the function where we want tracing to start

```python
def function_a(text_a):
    print(text_a)

@stacktrace
def function_b(text_b):
    print(text_b)
    function_a("text _a")

function_b('some text of function b')
```

Output:

> → Executing: function_b(‘some text of function b')
some text of function b
 → Executing: function_a(‘text _a')
text _a
 → Returning: function_a(‘text _a') -> None
 → Returning: function_b(‘some text of function b') -> None

### 4. Trace Class

Similar to the #3 **`@stacktrace`** we can define a decorator **`@traceclass`** which we use with classes, to get traces of its members' execution

```python
def traceclass(cls: type):
    def make_traced(cls: type, method_name: str, method: callable):
        def traced_method(*args, **kwargs):
            print(f'--> Executing: {cls.__name__}::{method_name}()')
            return method(*args, **kwargs)
        return traced_method
    for name in cls.__dict__.keys():
        if callable(getattr(cls, name)) and name != '__class__':
            setattr(cls, name, make_traced(cls, name, getattr(cls, name)))
    return cls

@traceclass
class SomeRandomClass:
    i: int = 0
    def __init__(self, i: int = 0):
        self.i = i
    def increment(self):
        self.i += 1
    def __str__(self):
        return f'This is a {self.__class__.__name__} object with i = {self.i}'

f1 = SomeRandomClass()
f2 = SomeRandomClass(4)
f1.increment()
print(f1)
print(f2)
```

### 5. Massive Parallel Processing

I use this decorator to deal with the situation where I need o run a function on all of my CPU threads in parallel. Things such as process embedding data or processing the same tasks over a bunch of documents.

I will put on an example to use all of your CPU to find primes. It is a simple one but you get the idea of massive and parallel processing data in real use case.

```sql

from random import seed, randint
from sympy import isprime
from joblib import Parallel, delayed
from time import time
from itertools import chain
from os import cpu_count


def parallel(func=None, args=(), merge_func=lambda x:x, parallelism = cpu_count()):
    def decorator(func: callable):
        def inner(*args, **kwargs):
            results = Parallel(n_jobs=parallelism)(delayed(func)(*args, **kwargs) for i in range(parallelism))
            return merge_func(results)
        return inner
    if func is None:
        return decorator
    else:
        return decorator(func)


@parallel(merge_func=lambda li: sorted(set(chain(*li))))
def generate_primes(domain: int=1000*1000, num_attempts: int=1000) -> list[int]:
    primes: set[int] = set()
    seed(time())
    for _ in range(num_attempts):
        candidate: int = randint(4, domain)
        if isprime(candidate):
            primes.add(candidate)
    return sorted(primes)

print(len(generate_primes()))
```

## Conclusion

In Python, functions are first-class citizens, and decorators are magically powerful wands to give programmers a seemingly "magic" way to construct useful compositions of functions and classes. By using wrappers, you have a pathway to simplifying intricate tasks, enhancing code readability, and boosting overall productivity.

In this article, we explored five examples

- Timer

- Retry

- Stacktrace

- Classtrace

- Parallel

The last three are complicated and way too advanced compared to the first two. Decorator is not too difficult to understand as you can see in the first two and you can really take your Python programming skills to the next level when you understand and utilize the last three.

---

If you do like this article, please give it a clap, seriously, a clap means a lot to me, and feel free to hit follow and subscribe to get notified for upcoming posts in the future. If you have any questions, please leave a comment, I may be slow to respond but I will try to answer as soon as possible.

If you need to reach out, don't hesitate to drop me a message via my [Twitter](https://twitter.com/kiennt_) or my [LinkedIn](https://www.linkedin.com/in/ryan-nguyen-abb844a4/).

**Subscribe to my substack as I cover more ML in-depth:** [https://howaibuildthis.substack.com/](https://howaibuildthis.substack.com/)