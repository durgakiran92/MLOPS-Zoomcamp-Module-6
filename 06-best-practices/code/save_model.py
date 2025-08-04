import mlflow.pyfunc


class DummyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return [42] * len(model_input)


# Save to a new directory (not the one with model.py inside)
mlflow.pyfunc.save_model(path="mlflow_model", python_model=DummyModel())
