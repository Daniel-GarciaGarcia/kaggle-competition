from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from time import time

def normalizador(df):
    pipeline = [    
        StandardScaler(),
        Normalizer(),]
    transformer = make_pipeline(*pipeline)
    X_new = transformer.fit_transform(df)
    return pd.DataFrame(X_new, columns=df.columns)

def numerador(df):
    for columna in df:
        
        if(df[columna].dtype == 'object'):
            df[columna] = df[columna].astype('category')
            df[columna] = df[columna].cat.codes
    return df


#probar modelos
def probarModelo(X,y,modelos):     
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)        
    
    for name,m  in modelos.items():        
        print(f"Entrenando {name}...")
        m = m.fit(X_train, y_train)
        print(m)
        print("Entrenamiento completo")      
        y_pred = m.predict(X_test)
        print(f"Evaluando modelo {name}")
        print(r2_score(y_test, y_pred))
        return y_pred




def optimizar_modelos(tuning,X ,Y):
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)  

    t0 =time()
    observando = tuning.fit(X_train,y_train)
    y_pred = observando.predict(X_test)
    print(r2_score(y_test, y_pred))

    print("Hecho en %0.3fs" % (time() - t0))
    print("Mejor  estimator encontado en la busqueda es:")
    print(observando.best_estimator_)
    return y_pred