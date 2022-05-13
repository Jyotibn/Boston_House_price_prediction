from flask import Flask,render_template,request
from flask_cors import CORS,cross_origin
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.ensemble import RandomForestRegressor

app=Flask(__name__)
@app.route('/',methods=['GET'])
@cross_origin()
def home_page():
    return render_template('index1.html')

@app.route('/predict1',methods=['GET','POST'])
def prediction():
    if request.method=='POST':
        try:
            boston=datasets.load_boston()
            df=pd.DataFrame(data=boston.data,columns=boston.feature_names)
            df['MEDV']=boston.target

            target_corr = df.corr()['MEDV']
            highly_correlated_feat = target_corr[abs(target_corr) >= 0.5]
            list_of_imp_feat=highly_correlated_feat.index[:-1]

            X=df[list_of_imp_feat]
            Y=df['MEDV']

            RM=float(request.form['RM'])
            PTRATIO=float(request.form['PTRATIO'])
            LSTAT=float(request.form['LSTAT'])


            x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=45)

            scaler=MinMaxScaler()
            x_train_scaled=pd.DataFrame(data=scaler.fit_transform(x_train,y_train),columns=list_of_imp_feat)

            x_test_scaled=pd.DataFrame(data=scaler.transform(x_test),columns=list_of_imp_feat)

            rfr=RandomForestRegressor(max_depth=8, max_features='sqrt', min_samples_split=5,n_estimators=200)
            rfr.fit(x_train_scaled,y_train)

            filename='final_random_forest.pickle'


            pickle.dump(rfr,open(filename,'wb'))
            loaded_model=pickle.load(open(filename,'rb'))
            predict=loaded_model.predict(scaler.transform([[RM,PTRATIO,LSTAT]]))

            return render_template('result1.html',predict=round(predict[0],2))

        except Exception as e:
            print("Exception occurred=",str(e))
            return "something is wrong"

    else:
        return render_template('index1.html')







if __name__=="__main__":
    app.run()