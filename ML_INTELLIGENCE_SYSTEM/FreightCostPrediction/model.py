from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_lr(xtrain,ytrain):
    model1=LinearRegression()
    model1.fit(xtrain,ytrain)
    return model1

def train_dtr(xtrain,ytrain,max_depth=4):
    model2=DecisionTreeRegressor(max_depth=4)
    model2.fit(xtrain,ytrain)
    return model2

def train_rfr(xtrain,ytrain,max_depth=4):
    model3=RandomForestRegressor(max_depth=4)
    model3.fit(xtrain,ytrain)
    return model3

def evaluate_model(model,xtest,ytest,model_name):
    pred=model.predict(xtest)
    mae=mean_absolute_error(ytest,pred)
    mse=mean_squared_error(ytest,pred)
    r2=r2_score(ytest,pred)*100

    print(f"\n{model_name}:Performance")
    print(f"MAE: {mae}")
    print(f"MSE:{mse}")
    print(f"R2:{r2:.2f}")

    return {
        "model":model_name,
        "mae":mae,
        "rmse":mse,
        "r2 score":r2
    }

    