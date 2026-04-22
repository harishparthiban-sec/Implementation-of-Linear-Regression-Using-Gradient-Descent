# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries: NumPy, Pandas, and Matplotlib.
2.Load the dataset and extract independent variable X (R&D Spend) and dependent variable Y (Profit).
3.Normalize the X values using mean and standard deviation.
4.Initialize parameters m (slope) and b (intercept), learning rate α, number of epochs, and dataset size.
5.Apply Gradient Descent: compute predicted values, calculate gradients, and update m and b iteratively.
6.Print final values of m and b, then plot the scatter graph and regression line.

## Program:

Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:  
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Startup.csv")
x=data['R&D Spend'].values
y=data['Profit'].values
x=(x-x.mean())/x.std()
m=0
b=0
alpha=0.01
epochs=1000
n=len(x)
for i in range(epochs):
    ynew=m*x+b
    dm=(-2/n)*np.sum(x*(y-ynew))
    db=(-2/n)*np.sum(y-ynew)
    m=m-alpha*dm
    b=b-alpha*db
print("Slope(m):",m)
print("Y-intercept(c):",b)
ynew=m*x+b
plt.scatter(x,y)
plt.plot(x,ynew)
plt.xlabel("R&D Spend(Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")
plt.show()
```

## Output:
<img width="999" height="635" alt="Screenshot 2026-04-22 093052" src="https://github.com/user-attachments/assets/cbb1a708-10c1-4e05-a8ab-b6714f16fb64" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
