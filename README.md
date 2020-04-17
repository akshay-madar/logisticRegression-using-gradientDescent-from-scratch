Here, I try to implement logistic regression using numpy. I have created functions for computing cost function and gradient descent. Please have a look at my personal notes below.

```
# computing cost function
def compute_cost(X, y, theta):
    m = len(y) # no of obs
    h = sigmoid(X.dot(theta)) # h = g(z) where z = theta.X 
    epsilon = 1e-5 # for computing non-zero log
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon))) # cost function to minimize
    return cost
    
# computing gradient descent
def gradient_descent(X, y, params, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros((iterations,1)) # for simultaneous updation

    for i in range(iterations): 
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y)) 
        cost_history[i] = compute_cost(X, y, params)

    return (cost_history, params)
```

![notes](https://github.com/akshay-madar/codestack/blob/master/Statistical%20Learning%20in%20Python/Logistic%20Regression%20using%20Gradient%20Descent/logistic_notes.png)
