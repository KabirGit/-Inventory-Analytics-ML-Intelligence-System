from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import cross_val_score

def train_random_forest(xtrain,ytrain):
    ytrain = ytrain.ravel()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": "balanced",
            "random_state": 42
        }

        model = RandomForestClassifier(**params)

        score = cross_val_score(
            model,
            xtrain,
            ytrain,
            cv=3,
            scoring="f1",
            n_jobs=-1
        ).mean()

        return score
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    best_model = RandomForestClassifier(**best_params, class_weight="balanced", random_state=42)
    best_model.fit(xtrain, ytrain)
    
    return best_model


def train_logistic_regression(xtrain,ytrain):
    model1=LogisticRegression(random_state=42)
    model1.fit(xtrain,ytrain)
    return model1
    

def evaluate_model(model, X_test, y_test,model_name):
    
    y_pred = model.predict(X_test)
    print("model name \n", model_name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))