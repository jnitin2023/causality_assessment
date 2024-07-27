import pandas as pd

# Create a realistic mock dataset
data = {
    'description': [
        'Patient experienced severe headache and nausea after taking medication A.',
        'No side effects observed with medication B after 1 month of use.',
        'Mild rash appeared on the arms and legs after the patient took medication C for a week.',
        'Patient reported severe dizziness and loss of balance after taking medication D for three days.',
        'No adverse effects noted with medication E even after two months of continuous use.',
        'Patient experienced shortness of breath and chest pain after the first dose of medication F.',
        'No side effects observed with medication G in the first week.',
        'Patient developed severe allergic reaction and skin rash after taking medication H.',
        'Patient reported fatigue and muscle weakness after taking medication I for two weeks.',
        'No adverse effects were reported with medication J during the initial trial period.'
    ],
    'causality': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]  # 1 indicates causality, 0 indicates no causality
}

df = pd.DataFrame(data)

# Save the mock dataset
df.to_csv('adverse_events.csv', index=False)
