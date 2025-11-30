import pandas as pd
import numpy as np
data=pd.read_csv("enjoysport2.csv")
df=pd.DataFrame(data)

prior_prob=df['EnjoySport'].value_counts(normalize=True)
print(f"Prior Probabilities:\n{prior_prob}\n")

df_yes=df[df['EnjoySport']=='Yes'] 
df_no=df[df['EnjoySport']=='No']
likelihood_yes = {
    column: df_yes[column].value_counts(normalize=True).to_dict()
    for column in ['Outlook', 'Temperature', 'Humidity', 'Wind']
}
likelihood_no = {
    column: df_no[column].value_counts(normalize=True).to_dict()
    for column in ['Outlook', 'Temperature', 'Humidity', 'Wind']
}

print("Likelihoods for Enjoysport=Yes")
for feature,values in likelihood_yes.items():
    for value,prob in values.items():
        print(f"{feature}={value}:{prob:.3f}")
        
print("Likelihoods for Enjoysport=No")
for feature,values in likelihood_no.items():
    for value,prob in values.items():
        print(f"{feature}={value}:{prob:.3f}")

def naive_bayes_classifier(test_instance,prior_prob,likelihood_yes,likelihood_no):
    posterior_yes=np.log(prior_prob['Yes'])
    posterior_no=np.log(prior_prob['No'])
    for feature,value in test_instance.items():
        posterior_yes+=np.log(likelihood_yes[feature].get(value,1e-6))
        posterior_no+=np.log(likelihood_no[feature].get(value,1e-6))
    print(f"posterior_yes:{posterior_yes}")
    print(f"posterior_no:{posterior_no}")
    return 'Yes' if posterior_yes > posterior_no else 'No'

test_instance={'Outlook':'Sunny','Temperature':'Cool','Humidity':'Normal','Wind':'Weak'}
prediction=naive_bayes_classifier(test_instance,prior_prob,likelihood_yes,likelihood_no)
print(f"Predicted class for test instance {test_instance}:{prediction})")

        
