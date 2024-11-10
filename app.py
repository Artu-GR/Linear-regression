from flask import Flask, render_template, request, session
from model import RegressionModel, Plotter

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secret key for session management

regression_model = RegressionModel('Database2.xlsx')

@app.route('/', methods=['GET', 'POST'])
def index():
    all_features = regression_model.features_info
    features_dict = {all_features[i]['title']:f"x{i+1}" for i in range(0, len(all_features))}
    inputs = {}
    predicted_value = ''
    
    #Retrieve the stored model and selected features from the session if they exist
    model = session.get('model', None)
    selected_features = session.get('selected_features', [])

    if request.method == 'GET':
        session.pop('model', None)
        session.pop('inputs', None)
        session.pop('selected_features', None)

    model = session.get('model', None)
    selected_features = session.get('selected_features', [])
    inputs = session.get('inputs', {})

    if request.method == 'POST':
        
        # Get selected features and corresponding input values
        form_selected_features = request.form.getlist('features')
        if form_selected_features:
            selected_features = form_selected_features
            session['selected_features'] = selected_features
        else:
            selected_features = form_selected_features
        inputs = {feature: request.form.get(f'input_{feature}', '') for feature in selected_features}
        print(inputs)
        # Check if "Generate Model" button is clicked
        if 'generate' in request.form:
            regression_model.fit(selected_features)
            model = f"y = {regression_model.coef_['Intercept']} + "
            model += " + ".join([f"({regression_model.coef_.get(feature, 0)})({features_dict[feature]})" for feature in selected_features])
            session['model'] = model

        # Check if "Predict Value" button is clicked
        if 'predict' in request.form and model:
            # Predict value based on the model and input values
            predicted_value = sum(
                float(inputs.get(feature, 0)) * regression_model.coef_.get(feature, 0)
                for feature in selected_features
                if inputs.get(feature).replace('.', '', 1).isdigit()  # Check for numeric input, including floats
            )
            predicted_value += regression_model.coef_['Intercept']
            predicted_value = round(predicted_value, 2)

        if 'plot' in request.form and model:
            plotter = Plotter(regression_model.coef_)
            print(selected_features)
            print("---------------------------------")
            plotter.update_plot(selected_features, 'Database2.xlsx')

    # Render the template and pass the necessary variables
    return render_template('index.html', all_features=all_features, selected_features=selected_features,
                           model=model, inputs=inputs, predicted_value=predicted_value, features_dict=features_dict)

if __name__ == "__main__":
    app.run(debug=True)
