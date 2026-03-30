import joblib
import numpy as np
import pandas as pd

model = joblib.load('./modelo_diabetes.pkl')
scaler = joblib.load('./scaler_diabetes.pkl')

print("="*45)
print("   PREDICTOR DE DIABETES")
print("="*45)
print("Ingresa los datos del paciente:\n")

pregnancies       = float(input("Número de embarazos:              "))
glucose           = float(input("Glucosa (mg/dL):                  "))
blood_pressure    = float(input("Presión arterial (mm Hg):         "))
skin_thickness    = float(input("Grosor de piel (mm):              "))
insulin           = float(input("Insulina (mu U/ml):               "))
bmi               = float(input("IMC (peso en kg/talla en m²):     "))
diabetes_pedigree = float(input("Función pedigree de diabetes:     "))
age               = float(input("Edad (años):                      "))

data        = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                       insulin, bmi, diabetes_pedigree, age]],
                     columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

datos_scaled = scaler.transform(data)
prediccion   = model.predict(datos_scaled)[0]
probabilidad = model.predict_proba(datos_scaled)[0][1]


print("\n" + "="*45)
print("   RESULTADO")
print("="*45)
if prediccion == 1:
    print(f"  Diagnóstico: DIABÉTICO")
else:
    print(f"  Diagnóstico: NO DIABÉTICO")
print(f"  Probabilidad de diabetes: {probabilidad:.1%}")
print("="*45)