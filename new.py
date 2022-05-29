import tkinter as tk
from tkinter import *
from tkinter import ttk, StringVar
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

root = Tk()
root.title('Machine Learning GUI')
root.geometry('1800x1800')

f1 = tk.Frame(root, bg="light green", highlightbackground="black", highlightthickness=1, width="650", height="780")
f1.place(x=0, y=0)

l_plot = Label(f1, text='Plot Various graphs using Car_sales CSV', font=('Helvetica', 15, "bold"), bg="light pink")
l_plot.place(x=100, y=15)


def data_1():
    global file_1
    file_1 = pd.read_csv(r"C:\Users\ajay_\PycharmProjects\Microsoft\car_sales.csv")
    for i in file_1.columns:
        if type(file_1[i][0]) == np.float64:
            file_1[i].fillna(file_1[i].mean(), inplace=True)
        elif type(file_1[i][0]) == np.int64:
            file_1[i].fillna(file_1[i].median(), inplace=True)
        elif type(file_1[i][0]) == type(""):
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            s = imp.fit_transform(file_1[i].values.reshape(-1, 1))
            file_1[i] = s
    colss = file_1.columns
    global X_Axis
    X_Axis = StringVar()
    X_Axis.set('X-axis')
    choose = ttk.Combobox(f1, width=22, textvariable=X_Axis, font=10)
    choose['values'] = (tuple(colss))
    choose.place(x=100, y=50)
    global Y_Axis
    Y_Axis = StringVar()
    Y_Axis.set('Y-axis')
    choose = ttk.Combobox(f1, width=22, textvariable=Y_Axis, font=10)
    choose['values'] = (tuple(colss))
    choose.place(x=100, y=100)
    global graphtype
    graphtype = StringVar()
    graphtype.set('Graph')
    choose = ttk.Combobox(f1, width=22, textvariable=graphtype, font=10)
    choose['values'] = ('scatter', 'line', 'bar', 'hist', 'corr', 'pie')
    choose.place(x=100, y=150)


# button to open the car sales csv
b0 = Button(f1, text="open", command=data_1, activeforeground="white", activebackground="black",
            font=('Helvetica', 15))
b0.place(x=550, y=15)


def plot():
    fig = Figure(figsize=(6, 6), dpi=70)
    global X_Axis
    global Y_Axis
    global graphtype
    u = graphtype.get()

    if u == 'scatter':
        plot1 = fig.add_subplot(111)
        plt.scatter(file_1[X_Axis.get()], file_1[Y_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(Y_Axis.get())
        plt.show()

    if u == 'line':
        plot1 = fig.add_subplot(111)
        plt.plot(file_1[X_Axis.get()], file_1[Y_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(Y_Axis.get())
        plt.show()

    if u == 'bar':
        plot1 = fig.add_subplot(111)
        plt.bar(file_1[X_Axis.get()], file_1[Y_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(Y_Axis.get())
        plt.show()

    if u == 'hist':
        plot1 = fig.add_subplot(111)
        plt.hist(file_1[X_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(X_Axis.get())
        plt.show()

    if u == 'corr':
        plot1 = fig.add_subplot(111)
        sns.heatmap(file_1.corr())
        plt.show()

    if u == 'pie':
        plot1 = fig.add_subplot(111)
        plt.pie(file_1[Y_Axis.get()].value_counts(), labels=file_1[Y_Axis.get()].unique())
        plt.show()


b1 = Button(f1, text="Plot", command=plot, activeforeground="white", activebackground="black",
            font=('Helvetica', 15, 'bold')).place(x=100, y=250)
# function to open pre-processed file (refer the code in cars.py)
def data():
    global file
    file = pd.read_csv(r"C:\Users\ajay_\PycharmProjects\Microsoft\pre_processed_file.csv")
    for i in file.columns:
        box1.insert(END, i)
    for i in file.columns:
        if type(file[i][0]) == np.float64:
            file[i].fillna(file[i].mean(), inplace=True)
        elif type(file[i][0]) == np.int64:
            file[i].fillna(file[i].median(), inplace=True)
        elif type(file[i][0]) == type(""):
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            s = imp.fit_transform(file[i].values.reshape(-1, 1))
            file[i] = s
    colss = file.columns


# Function to input and read x from list box
def getx():
    global s
    x_v = []
    s = box1.curselection()
    global feature_col
    for i in s:
        if i not in feature_col:
            feature_col.append(file.columns[i])
            x_v = feature_col
    for i in x_v:
        box2.insert(END, i)


# Function to input and read x from list box
def gety():
    y_v = []
    global target_col
    s = box1.curselection()
    for j in s:
        if j not in target_col:
            target_col.append(file.columns[j])
            y_v = target_col
    for i in y_v:
        box3.insert(END, i)


# variables to store x and y values
feature_col = []
target_col = []


# function to predict values
def model():
    x = file[feature_col]
    y = file[target_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    parameters = {"n_estimators": list(range(10, 50, 5))}
    regressor = LinearRegression()
    regressor.fit(x, y)
    x_dummies = s
    x_tests = []
    y_pred = regressor.predict(x_test)
    accuracy = r2_score(y_test, y_pred)
    Label(f2, text="Accuracy", font=('Helvetica', 15, 'bold'), bg="light green").place(x=100, y=550)
    Label(f2, text=str(accuracy), font=('Helvetica', 15, 'bold'), bg="light green").place(x=200, y=550)
    b0 = regressor.coef_
    b1 = regressor.intercept_
    x_unknown = tk.DoubleVar()

    # function to calculate the estimated value
    def calculate():
        global y_expected
        e1 = tk.Entry(f2, font=("aria", 15, "bold"), textvariable=x_unknown, width=8, bg="white")
        e1.place(x=200, y=600)
        x_1 = float(x_unknown.get())
        if x_1 > 0:
            y_expected = b0 + b1 * x_1
            print(x_1)
            print()
            Label(f2, text=str(y_expected), font=('Helvetica', 15, 'bold'), bg="light green").place(x=200, y=700)
        else:
           Label(f2, text="X can't be negative", font=('Helvetica', 15, 'bold'), bg="light green").place(x=200, y=700)
        if y_expected<0:
            Label(f2, text="Selected features are negatively corelated", font=('Helvetica', 15, 'bold'), bg="light green").place(x=200, y=730)

    b_cal=Button(f2, text='calculate', command=calculate, activeforeground="white", activebackground="black",
           font=('Helvetica', 15)).place(x=100, y=650)

    Label(f2, text="Enter X", font=('Helvetica', 15, 'bold'), width=8, bg="light green").place(x=100, y=600)
    Label(f2, text="Predicted Y", font=('Helvetica', 15, 'bold'), bg="white").place(x=100, y=700)
    return x_tests, y_pred


# frame to display the predicte
f2 = tk.Frame(root, bg="light green", highlightbackground="black", highlightthickness=1, width="800", height="780")
f2.place(x=700, y=10)

# labels used:
l_predict = Label(f2, text='Regressor to estimate ', font=('Helvetica', 15, "bold"), bg="light pink")
l_predict.place(x=100, y=15)

l1 = Label(f2, text='data file', font=('Helvetica', 20), bg="white")
l1.place(x=80, y=55)

l2 = Label(f2, text="Cars Sales", font=('Helvetica', 20), bg="white")
l2.place(x=240, y=55)

# display list of columns
box1 = Listbox(f2, selectmode='single', font=10)
box1.place(x=100, y=200)

# display selected x
box2 = Listbox(f2, font=10)
box2.place(x=300, y=200)

# display selected y
box3 = Listbox(f2, font=10)
box3.place(x=500, y=200)

# buttons used in frame:2
b2 = Button(f2, text='display', command=data, activeforeground="white", activebackground="black",
            font=('Helvetica', 15))
b2.place(x=400, y=55)

b3 = Button(f2, text='Select X', command=getx, activeforeground="white", activebackground="black",
            font=('Helvetica', 15))
b3.place(x=300, y=450)

b4 = Button(f2, text='Select Y', command=gety, activeforeground="white", activebackground="black",
            font=('Helvetica', 15))
b4.place(x=500, y=450)

b5 = Button(f2, text="predict", command=model, activeforeground="white", activebackground="black",
            font=('Helvetica', 15))
b5.place(x=100, y=500)

root.mainloop()
