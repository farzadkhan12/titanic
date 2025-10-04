from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import re

class PreProcer(BaseEstimator, TransformerMixin):
    def fit(self,X, y=None):
        self.ageimpute = SimpleImputer()
        self.ageimpute.fit(X[["age"]])
        return self
    def transform(self,X, y=None):
        X["age"] = self.ageimpute.transform(X[["age"]])
        X["cabinClass"] = X["cabin"].fillna("M").apply(lambda x:str(x).replace(" ","")).apply(lambda x: re.sub(r"[^a-zA-Z]", '',x)).values
        X["cabinNumber"] = X["cabin"].fillna("M").apply(lambda x:str(x).replace(" ","")).apply(lambda x: re.sub(r"[^0-9]", '',x)).replace("",0).values
        X["embarked"] = X["embarked"].fillna("M")
        X = X.drop(["name", "ticket", "cabin", "boat", "body","home.dest"], axis=1, errors="ignore")
        return X

columns = ['name','sex','pclass','ticket','sibsp','age','parch','fare','cabin','embarked']