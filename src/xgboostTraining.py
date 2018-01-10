import xgboost as xgb

def trainXGModel(train, verify):
    target = train['unit_sales']
    train.drop(['unit_sales'], axis=1)   
    xgtrain = xgb.DMatrix(train.values, target)
    
    xgModel = xgb.train()    
    return xgModel
