This is an outline of a talk I originally gave at the Center for Computational Neuroscience at the Flatiron Institute, on August 5th, 2025.

---

Welcome to *Python, Rust, and You: Modern Py-Rust Interoperation*!

Please open up your preferred way to execute python: your terminal, Jupyter Notebook, python script, what have you.

If you have the Rust toolchain installed, you may also clone this repository directly and follow along through each commit one at a time.

---

Before we start, who here has used a Python package backended in Rust before? Raise your hands.

That's good to see. Now, please run the following commmands in your Python interface of choice:

Terminal: 
```
pip install scrutipy

python3

from scrutipy import closure

results = closure(3.5, 1.2, 50, 0, 7, 0.05, 0.005)

len(results)

exit
```

Jupyter/Script
```
!pip install scrutipy

import scrutipy
from scrutipy import closure

results = closure(3.5, 1.2, 50, 0, 7, 0.05, 0.005)

len(results)
```

Congratulations, you have all now used a Python package backended in Rust!

Specifically, this is my PyPI package, ScrutiPy, a small library for scientific error detection. The function you've just used is an implementation of the novel CLOSURE algorithm by Nathanael Larigaldie, which is designed to collect all the possible datasets that could have generated a set of summary data, to within stated rounding and tolerances.

This function has two main desiderata: performance and ease of use.

It must be performant because this is a combinatorially explosive problem: the toy version you ran just now, which I use in the documentation as a quick example, must find 7980 datasets out of a possible 1.4e45 grid points. We cannot afford to leave any algorithmic or computational opimizations on the table.

It must be easy to use because it is not aimed at computer scientists or professional programmers: it is aimed at researchers, journal editors, and peer reviewers, primarily in the medical and social sciences. We do not assume that our target audience has the technical expertise or patience to debug their installation if it does not work immediately after a one-line install. If it doesn't work out of the box, it will not be used.

Moreover, we need to support the use of this algorithm in both Python and R, since our target audience tends to use one of the two languages, but not both.

To satisfy both of these conditions, we chose Rust as the core of our library. I maintain the Python frontend, while my colleague Lukas Jung at the University of Bern maintains the R frontend as part of the `unsum` CRAN library.

Given Rust's reputation as a low-level and difficult language, you may be surprised to hear it mentioned in the same breath as 'ease of use'. But we shall soon find that one of Rust's many virtues is its powerful support and integration into many languages and platforms, such that we prefer it over more common and similarly performant languages such as C. 

To demonstrate this, we will go through the motions of building a small Rust-Python library that demonstrates many features of the problem and also provides good opportunities for high-performance benchmarking: in this case, matrix exponentiation.

For the optimal experience, I recommend using the `jujutsu` VCS, which allows you to easily move backwards and forwards between git commits. However, you can also move between git commits manually if you wish.

---

We will first go through the steps of creating a Rust-Python library project:

Step 0: Install Rust and `maturin`
One does, of course, need to have [Rust installed](https://www.rust-lang.org/tools/install) in order to develop in Rust. If you are installing it for the first time, it is also highly recommended that you install Cargo. 
In addition, we will use the `maturin` crate to organize our code and provide bindings. Full instructions on how to install `maturin` can be found on its [user guide](https://www.maturin.rs/installation.html). Note that installing `maturin` with `brew` or similar installers will also install Rust.

Step 1: In your project directory of choice, run `maturin new coding_club_example`. You will be prompted to choose your bindings: choose PyO3 
Step 2: Run `cd coding_club_example` to enter this maturin project
Step 3: Run `python3 -m venv .venv` to create a virtual environment in the project. You may also use `uv` or your other environment manager of choice.
Step 4: Run `source .venv/bin/activate` to activate the virtual environment 
Step 5: Run `pip install -U pip maturin` to ensure that your environment contains the most up-to-date versions of both pip and maturin 

Congratulations, you've set up your first Rust-Python project!

If you explore your project now, you will see that it contains a Rust library file (lib.rs), several Cargo files for package management, and a pyproject.toml file. For the most part, these will be handled automatically, without much need to edit them ourselves.

---

### Initial Commit 
Now, let's begin stepping through the commits one by one. The first commit, titled 'initial commit' contains the same basic environment you have just created locally.

Here, we also see the structure of a Rust-Python project in miniature: it has created by default two functions: a `sum_as_string` function, which is marked with the `#[pyfunction]` decorator, and a `coding_club` function, which is marked by the `#[pymodule]` decorator.

Whenever we want to call a Rust function from Python, we will add the `#[pyfunction]` decorator, and we will have to ensure that its inputs can be translated from Python to Rust, and that its outputs can be translated from Rust to Python. For many input and output types, this is very simple, but for the sake of demonstration we will be using some more complicated types that demonstrate the flexibility of this approach. 

When we want to add a function to our library, we simply add it to the `#[pymodule]`, for which there may only be one per project.

### Added Dependencies
Let's move to the next commit, titled 'added dependencies'. Here we have added the only two Rust libraries that our code will depend on: `num` and `ndarray`. `num` is a Rust library that provides useful support for generic numeric types, and `ndarray` is a popular Rust library for matrix operations, including the dot product function that we will be using.

### Added Function Signatures
We will now outline the flow of our functions by defining their headers: you may have seen this kind of notation if you've used type python type checkers such as `mypy`, `ty`, or `pyrefly`. Unlike Python, which is much looser with its types, Rust strictly requires that all inputs and outputs of a function have their types defined up-front.

In this case, we've defined three things: 
- A custom `Matrix<A>` type, where `A` means any numeric type that can be used for linear algebra operations (floats, signed and unsigned integers, and so on)
- A function exclusively using native Rust, which takes in a `Matrix<A>` and `usize` (a 'pointer-sized' unsigned integer which is either 32 or 64 bits depending on the computer architecture running it). This function will do the computational heavy lifting.
- And a pyfunction which will bridge the gap between Python types and Rust types. It takes as inputs a `&Bound<'_, PyAny>` and a `usize`. The first of these types looks rather strange and esoteric, but this PyO3's way of saying 'look, Python could pass absolutely anything into this argument' and telling Rust's type checker not to worry about it: we'll handle it manually. 

The advantage of this approach is that it gives us a lot of flexibility on the Python user's side. Do they want to pass us some nested arrays with floats? Nested arrays with integers? A list of arrays of floats and/or integers? A nested list of floats and integers all mixed together? Doesn't matter. So long as it's a valid Python object that we can convert into an ordered array (so, not a set or dictionary), we can pass it into the same function with zero worries.

### Adding Matrix Exponentiation Algorithm
We'll now add in the core of this library: the matrix exponentiation algorithm itself, written entirely in native Rust. Brace yourselves...

Not too scary, right? 

This is a basic exponentiation algorithm that lets us find the final answer in logN time, and other than the `let`s and `mut`s, this doesn't read very differently from Python. The `mut` is just us telling Rust that the `result` and `base` objects can be changed after they're initially declared: Python lets you do that to anything at any time, but in Rust, things are immutable after their creation unless you say otherwise!

### Extract and Transform

We're not going to touch the Rust helper function again from this point on: we're just going to process whatever data Python gives us into a format that our Rust-only function can use without complaining.

First, we're going to take the matrix object that we've taken as an input and attempt to `extract()` it. PyO3 will attempt to transform this object into another type: in this case, a nested array of floats (`Vec` is Rust's built-in, heap-allocated array). So long as our input consistents of a 2d nested array-like object containing numbers: lists, arrays, floats, integers, etc, this should succeed.

If it does not succeed, we will exit with an error. We do this with the `?` notation at the end of the line, which is shorthand for 'if this succeeds, return the result, otherwise exit with an error'. This allows us to directly exit with a Python TypeError, so a Python-side user sees an error of the kind they're familiar with.

From here, we get the dimensions of the new nested_vec object in order to transform it into our `Matrix<A>` type. Here you can see an example of manual error handling, to catch the empty matrix case. Note that we're manually creating a Python ValueError type and returning it: once again, we want to raise Errors in a way which is customary and informative to a Python user.

Finally, we flatten the nested vectors in to a single, 1-d vector. This is very similar to numpy's `np.flatten()` method, with a couple notable differences: instead of directly calling `.flatten()`, we use `.into_iter()` and `.collect()`. This is the iterator pattern, which is very common in Rust: instead of just using `for n in nested_vecs`, which would also work in Rust, we transform an iterable type like a Vector into an `iterator`, which is a bit like a range that operates on all the elements of the iterable. We can then do whatever we want with these elements (in this case, flatten them), and then `collect()` the output into a new iterable: in this case, a 1-d vector.

The explicit use of iterators is a language feature taken from functional programming, and it is virtually always faster, often much faster, than a regular for loop in Rust. Note also that because we used `.into_iter()` instead of `.iter()`, we consume the original nested_vecs value: from this line onwards, nested_vecs no longer exists in our program!

### Matrix from Vec

Okay, that was a lot to take in, but it's all smooth sailing from here.

Now we're just creating a 2-d array using our newly flattened vector and the shape of the input matrix. We use the Array2 object's build-in-method to do this, but we handle the error case ourselves: if the dimensions of the vector don't match what we passed in, i.e. it can't be turned into a square matrix of size nxn, we return a Python ValueError, as before.

### Call Rust Function

Now that we have everything in the desired types, we just call our Rust matrix exponentiation function directly: no sweat!

### Transform Result

Now we're taking the matrix we got out and turning it back into a nested vector: we get the rows (which are an iterable), turn them into an iterator, turn each row into a separate vector, and then collect them into a vector of vectors. Done!

### Return Output

Finally, we take this output and return it: we wrap it in `Ok()` to signal to the Rust compiler that this is a desired output, not an error.

And... we're done! 

This was a lot to take in, but if you advance to the next commit, `removing comments`, you'll see that without the explanatory comments... this isn't all that much code. In just about 50 lines, we handle every step of the process, from transforming the inputs, performing the computation, and then transforming the outputs. 

But how do we actually use this? 

### Developing with `maturin`

The steps of actually compiling and using this code are very straightforward. We just use one line:

`maturin develop -r`

Note that to use this, you need to be in the environment you created before.

The `develop` command compiles our Rust code into a Python wheel: this wheel, called `coding_club`, makes all the functions we defined in the pymodule to the end user.

The -r flag is short for --release. If you omit it, then the process of developing will be faster, but the code won't run as fast: by default, Rust compiles `debug` builds which compile more quickly but aren't as heavily optimized. This helps us save time during development, but right now, we're done with development and want maximum speed at runtime, which `release` builds give us. 

### Using the wheel

Within the development environment, open the `benchmark.ipynb` Jupyter notebook: inside, you will find several functions for benchmarking our new Rust-Python code against both NumPy's `linalg.matrix_power()` function and a native Python implementation of the same algorithm. Note that you'll need to `pip install numpy` into the same environment!

Because the coding_club wheel is already present in the environment, we don't need to do anything special: just import coding_club as if it were any other Python package you downloaded with `pip`. 

Note that the coding_club.whl wheel is just a file like any other: you can copy it, send it to another folder or environment, distribute it on PyPI, or just send it in an email. It was built with Rust and maturin, but as you saw with the `scrutipy` wheel earlier, you don't need to have either of those installed to run the program as a user. There also isn't any dependency on other Python packages like numpy or tensorflow, or a parallelization backend like OpenBLAS (though we could enable BLAS in our `ndarray` install to optimize it if we wanted to). As far as the notebook, or any other way of running Python code is concerned, this might as well have been written in native Python with zero dependencies, but just happens to be several times faster. 

This is one of the major promises of using Rust for Python backends: the only things you need to run it are an operating system and a supported version of Python. So long as a wheel exists for your combination of major Python version and OS, it will run with no other installations, requirements, or version conflicts. This means you can add a package with a Rust backend into your existing codebase just by installing it, and can be confident it won't interfere with the rest of your build.

This is the same level of user convenience delivered by packages like NumPy (which are backended manually in CPython, much harder than writing either native Python or native C), but much, much easier to develop and maintain.

### Benchmarking

You can run through the cells one by one: note that the benchmarking code will take several minutes to run.

Assuming nothing has changed dramatically since the time of writing, we should see that the native Python code is slowest, NumPy is fastest, and our Python-Rust code is in the middle, though it's closer in performance to NumPy than Python.

It's no surprise that Rust is faster than Python, though maybe it surprises you just how much faster: but why is Numpy still 8-10x faster?

The simple reason: NumPy is one of the most heavily optimized computing libraries in the world, and the linalg module in particular is extremely heavily optimized, since it is used at large scale for machine learning operations which do nothing but crunch matrices all day long. We should not expect some Rust code that we whipped up in a few minutes to have comparable performance, especially since we haven't even turned on `ndarray`'s BLAS backend.

The other reason is that, because we are crossing the Python-Rust FFI, there's a small amount of overhead that does not exist for NumPy, which benefits from being backended directly in CPython. 

This brings us to some guidelines and takeaways for Python-Rust development. 

First: if NumPy already does what you need, you probably won't get any benefit out of writing a Rust version instead. You'll be spending extra time for the privilege of being slower.

Second: the primary targets for Rust optimization are operations which existing, optimized libraries do not already perform.

### Real Example: Cubic roots and Simulations

Let's take an actual example of some optimizations I've been working on recently. The source for this is a scientific simulation codebase.

```
def cubic_y_root(x0: float, y0: float) -> [float]:
    """Calculate the root of cubic function f(y) = x0*y^3 + 1.5*y - y0
    Returns the roots of the function
    """

    coefficients = np.array([x0, 0, +1.5, -y0])
    poly = np.polynomial.Polynomial(coefficients[::-1])
    return poly.roots()
```

This is a very brief cubic root solver, taking advantage of NumPy's powerful and well-optimized polynomial.roots() function. 

So, this would be an example of a function that should be left alone and not optimized with Rust, right?

As it turns out, no.

This is because NumPy's polynomial root solver is a very powerful, general-use eigenvector solver: this is great if you need to solve arbitrary polynomials for orders p > 4, for which closed-form solutions do not exist. But if you're solving cubic roots, and in fact, are solving the exact same cubic root each time, an eigenvector solver is not only totally overpowered, the repeated setup cost of initializing it makes it slower than a naive analytical solver!

We can sketch out a native-Python analytical solver as follows: 

```
def cubic_y_root_cardano(x0: float, y0: float)  -> [float]:
    """
    Optimized version of cubic_y_root using an analytic solver.
    Solves the equation: x0*y^3 + 1.5*y - y0 = 0
    """
    # handle the edge case where x0 is zero, becomes 1.5*y - y0 = 0
    if x0 == 0:
        return np.array([y0 / 1.5])

    # convert to the standard depressed cubic form y^3 + p*y + q = 0
    # by dividing the original equation by the leading coefficient x0
    p = 1.5 / x0
    q = -y0 / x0

    # calculate the discriminant term to see if there will be one or three real roots
    delta = (q/2)**2 + (p/3)**3

    if delta >= 0:
        # discriminant positive or 0, one real root, two complex roots
        sqrt_delta = np.sqrt(delta)
        u = np.cbrt(-q/2 + sqrt_delta)
        v = np.cbrt(-q/2 - sqrt_delta)
        roots = np.array([u + v])
    else:
        # discriminant negative, three real roots
        term1 = 2 * np.sqrt(-p / 3)
        phi = np.arccos((3 * q) / (p * term1))
        
        y1 = term1 * np.cos(phi / 3)
        y2 = term1 * np.cos((phi + 2 * np.pi) / 3)
        y3 = term1 * np.cos((phi + 4 * np.pi) / 3)
        roots = np.array([y1, y2, y3])
        
    return roots
```

It's longer and wordier... but it's also up to 9x faster than the poly root solver, with absolutely no Rust, C, or CPython involved! 

This is a more general takeaway for optimization (though Casey Muratori would term this 'de-pessimization' instead): when you use someone else's optimized functions, make sure that you know what it's doing under the hood! It doesn't matter how optimized NumPy is if it winds up spending thousands of cycles spinning its `.whl`s on unnecessary work.

Alright, but since we're using lots of numpy functions internally here for trigonometric and array operations, this means that we shouldn't bother with a Rust version, right? 

Once again, no. Those individual trigonometric operations are quite well optimized as far as Python goes, but they're still being orchestrated by an interpreted language, and that contributes to a ton of overhead in addition to losing compile-time optimizations.

Let's outline a Rust pyfunction for the above:

```
#[pyfunction(signature = (x0, y0))]
pub fn cubic_y_root_cardano(x0: f64, y0: f64) -> ([f64; 3], usize) {
    if x0 == 0.0 {
        ([y0/1.5, 0.0, 0.0], 1) //padding out the array
    } else {
        let p = 1.5/x0;
        let q  = -y0/x0;

        let delta = (q/2.0).powi(2) + (p/3.0).powi(3);

        if delta > 0.0 {
            // one real root path
            let sqrt_delta = delta.sqrt();
            let u = (-q/2.0 + sqrt_delta).cbrt();
            let v = (-q/2.0 - sqrt_delta).cbrt();
            ([u + v, 0.0, 0.0], 1) //padding out the array
        } else {
            // three real roots path
            let term1 = 2.0 * (-p / 3.0).sqrt();
            let phi = ((3.0 * q) / (p * term1)).acos();

            let y1 = term1 * (phi / 3.0).cos();
            let y2 = term1 * ((phi + 2.0 * PI) / 3.0).cos();
            let y3 = term1 * ((phi + 4.0 * PI) / 3.0).cos();
            ([y1, y2, y3], 3)
        }
    }
}
```

You'll notice that this is even simpler than the matrix example from earlier: because floats are the only input and output, we don't need to do any transformations: working with primitive types is much easier, and should be preferred whenever possible.

Second, you might also notice that this implementation is totally heapless: we perform all operations on the stack using fixed-size arrays, and return a fixed-size array along with a counter of how many roots are real. This allows us to trade memory for runtime performance, using a little more memory than necessary to avoid heap allocation calls. This is the kind of fine control that is readily available in low level languages like C and Rust, but is generally not available in native Python.

The results:

Polynomial.roots:         15.80 microseconds/call  1.0x

Python (Analytical):       1.77 microseconds/call  8.92x

Rust-Python (Analytical):  0.37 microseconds/call  46.47x

On some quick-and-dirty benchmarks, our analytical implementation improved almost 9x over Polynomial.roots(), and our Rust implementation was nearly 7x faster *again* on top of that, nearly 50x faster than the original.

Now, it must be said that benchmarking languages like this is a difficult art, and my benchmarks here are not the most elaborate or complete. But the real test is in application, and in this case, our optimizations proved true.

See, this code originated in a large, CPU-bound galaxy simulation codebase: when benchmarked, it was found to be spending more than half of its total runtime in the body of just one simulation function: and the inner loop of that simulation function was dominated by these polynomial root calculations.

Benchmarking that function as a whole, switching from polynomial solvers to analytic solvers (both the one detailed above and another, similar one) resulted in a 1.3-1.6x increase in runtime performance compared to the original. The Rust-Python version benchmarked a 3.4-4.8x increase in performance compared to the original. The former saved us approximately a sixth of total runtime, and the latter saved us approximately half our total runtime.

The microseconds add up.

---

Thank you for reading and following along with *Python, Rust, and You: Modern Py-Rust Interoperation*! If you enjoyed it, please give the repo a star and share it in your communities.
