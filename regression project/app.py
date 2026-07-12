from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    equation = ""
    graph = False
    x = ""
    y = ""

    mse = ""
    mae = ""
    rmse = ""
    r2 = ""

    if request.method == "POST":

        x = request.form["x"]
        y = request.form["y"]

        X = np.array([float(i) for i in x.split(",")]).reshape(-1,1)
        Y = np.array([float(i) for i in y.split(",")])

        # Linear Regression Model
        model = LinearRegression()
        model.fit(X, Y)

        y_pred = model.predict(X)

        # Equation
        m = model.coef_[0]
        c = model.intercept_

        equation = f"Regression Line : y = {m:.2f}x + {c:.2f}"

        # Evaluation Metrics
        mse = round(mean_squared_error(Y, y_pred),4)
        mae = round(mean_absolute_error(Y, y_pred),4)
        rmse = round(np.sqrt(mse),4)
        r2 = round(r2_score(Y, y_pred),4)

        action = request.form["action"]

        if action == "visualize":

            if not os.path.exists("static"):
                os.makedirs("static")

            plt.figure(figsize=(8,10))

            # ---------------- Graph 1 ----------------
            plt.subplot(2,1,1)

            plt.scatter(X, Y, color="blue", label="Actual Data")
            plt.plot(X, y_pred, color="red", linewidth=2, label="Best Fit Line")

            plt.title("Linear Regression")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.grid(True)

            # ---------------- Graph 2 ----------------
            plt.subplot(2,1,2)

            metrics = ["MSE","MAE","RMSE","R²"]
            values = [mse, mae, rmse, r2]

            plt.scatter(metrics, values, color="purple", s=100)

            for i,value in enumerate(values):
                plt.text(i, value+0.02, f"{value:.4f}",
                         ha="center", fontsize=10)

            plt.title("Model Metrics Comparison")
            plt.xlabel("Evaluation Metrics")
            plt.ylabel("Values")
            plt.grid(True)

            plt.tight_layout()

            plt.savefig("static/graph.png")
            plt.close()

            graph = True

    return render_template(
        "index.html",
        equation=equation,
        graph=graph,
        x=x,
        y=y,
        mse=mse,
        mae=mae,
        rmse=rmse,
        r2=r2
    )

if __name__ == "__main__":
    app.run(debug=True)