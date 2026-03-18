import joblib
from pathlib import Path

from DataPreprocessing import load_vendor_invoice,split_data,prepare_features
from model import train_lr,train_dtr,train_rfr,evaluate_model


def main():
    db_path="C:\\Users\\Kabir\\Desktop\\my SALES ANALYSIS\\data\\inventory.db"
    model_dir=Path("models")
    model_dir.mkdir(exist_ok=True)

    #load data
    df=load_vendor_invoice(db_path)

    #prepare and split data
    x,y=prepare_features(df)
    xtrain,xtest,ytrain,ytest=split_data(x,y)

    #lets train the modelllls
    model1=train_lr(xtrain,ytrain)
    model2=train_dtr(xtrain,ytrain)
    model3=train_rfr(xtrain,ytrain)

    #evaluate the model
    results=[]
    results.append(evaluate_model(model1,xtest,ytest,"linear regression"))
    results.append(evaluate_model(model2,xtest,ytest,"Decision tree regressor"))
    results.append(evaluate_model(model3,xtest,ytest,"Random forest regressor"))

    #choosing the best model
    best_model_info=min(results,key=lambda x:x["mae"])
    best_model_name=best_model_info["model"]

    best_model={
        "linear regression":model1,
        "Decision tree regressor":model2,
        "Random forest regressor":model3
        
    }[best_model_name]

    #saving the best model
    model_path=model_dir/"predict_freight_cost_model.pkl"
    joblib.dump(best_model,model_path)

    print(f"best model saved:{best_model_name}")
    print(f"model path:{model_path}")

if __name__=="__main__":
    main()










