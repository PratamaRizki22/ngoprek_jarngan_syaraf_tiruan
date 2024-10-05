class Perceptron:
    def __init__(self, learning_rate=0.575, threshold=0.555, initial_weights=None, initial_bias=0):
        self.lr = learning_rate
        self.activation_func = self._unit_step_function
        self.threshold = threshold
        self.weights = initial_weights 
        self.bias = initial_bias       

    def _unit_step_function(self, x):
        return np.where(x >= self.threshold, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        if self.weights is None:
            self.weights = np.zeros(n_features)

        epoch = 0

        while True:
            epoch += 1
            print(f"\n=== Epoch {epoch} ===")
            wrong = 0  # Counter untuk prediksi salah
            oke = 0    # Counter untuk prediksi benar

            for idx, x_i in enumerate(X):
                # Hasil dari operasi linear (dot product)
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                print(f"\nSample {idx + 1}:")
                print(f"    Input: {x_i}")
                print(f"    Bobot saat ini: {self.weights}")
                print(f"    Bias saat ini: {self.bias}")
                print(f"    Hasil linear (dot product): {linear_output}")
                print(f"    Prediksi: {y_predicted}, Target: {y[idx]}")

                update = self.lr * (y[idx] - y_predicted)
                print(f"    Delta (y_true - y_pred): {y[idx] - y_predicted}")
                print(f"    Pembaruan bobot (Delta w): {update} = {self.lr} * ({y[idx]} - {y_predicted})")

                self.weights += update * x_i
                self.bias += update

                print(f"    Bobot setelah pembaruan: {self.weights}")
                print(f"    Bias setelah pembaruan: {self.bias}")

                if y_predicted == y[idx]:
                    oke += 1
                else:
                    wrong += 1

            print(f"\nOke (benar): {oke}, Wrong (salah): {wrong}")

            if wrong == 0:
                print(f"\nTraining selesai di epoch {epoch}")
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

X = np.array([[1, 1], [1, 0], [0,1], [0, 0]])
y = np.array([1, 1, 1, 0])  # AND logic gate

initial_weights = np.array([5.1, -0.75]) 
initial_bias = 1.5 

perceptron = Perceptron(learning_rate=0.575, threshold=0.555, initial_weights=initial_weights, initial_bias=initial_bias)
perceptron.fit(X, y)

predictions = perceptron.predict(X)
print(f'Predictions: {predictions}')
