import pandas as pd
import statsmodels.api as sm
import plotly.graph_objs as go
import plotly.offline as pyo
from sklearn.model_selection import train_test_split

class RegressionModel:
    def __init__(self, data_file):
        # Load and clean the data once
        self.df = pd.read_excel(data_file)
        self.df_cleaned = self.df.dropna()

        # Prepare the dependent variable
        self.y = self.df_cleaned['Consumo de energia electrica (kWh per capita)']
        
        self.features_info=[
                    {
                        'title': 'Emisiones de C02',
                        'description': 'Hace referencia a la cantidad de CO2 generado por combustibles fosiles y la industria, medido en toneladas.',
                        'mexico_info': 'Tan solo en 2019, Mexico emitio mas de 700 toneladas de CO2 posicionandose como el 13avo pais con mayores emisiones de CO2 del mundo.',
                        'image': 'https://cdni.iconscout.com/illustration/premium/thumb/residuos-de-fabricas-de-productos-quimicos-10773350-8662142.png',
                        'uom':'Toneledas per capita'
                    },
                    {
                        'title': 'Control de corrupcion',
                        'description': 'Capta las percepciones de hasta que punto el poder publico se ejerce para beneficio privado, incluyendo tanto las pequenas como las grandes formas de corrupcion, asi como la "captura" del Estado por parte de las elites y los intereses privados. Su valor va de -2.5 hasta 2.5',
                        'mexico_info': 'Mexico ha ido en decadencia pasando de un -0.07 en el ano 2000, a un -1.01 en el ano 2022.',
                        'image': 'https://advokatnidenik.cz/wp-content/uploads/lobbying-161689_960_720-4.png',
                        'uom':'Coeficiente'
                        
                    },
                    {
                        'title': 'Personas que usan el internet',
                        'description': 'Los usuarios de Internet son personas que han utilizado Internet (desde cualquier lugar) en los ultimos 3 meses. ',
                        'mexico_info': 'Se estima que en Mexico alrededor de 100 millones de personas usan internet, convirtiendose en el segundo pais latinoamericano con mayor poblacion internauta.',
                        'image': 'https://www.ovhcloud.com/sites/default/files/styles/large_screens_1x/public/2022-04/whatis_internet-of-things.png',
                        'uom':'% de la poblacion'
                    },
                    {
                        'title':'Empleados en el sector industrial',
                        'description': 'Personas con edad para trabajar que participan en el sector industrial, que esta compuesto por la mineria y la explotacion de canteras, la manufactura, la construccion y los servicios publicos (electricidad, gas y agua)',
                        'mexico_info': 'Se estima que en Mexico existen alrededor de 10 millones de personas trabajando en la industria.',
                        'image': 'https://static.vecteezy.com/system/resources/previews/027/182/827/non_2x/industrial-plant-isolated-on-a-transparent-background-oil-and-gas-industry-refinery-factory-petrochemical-plant-area-png.png',
                        'uom':'% de los empleos'
                    },
                    {
                        'title':'Gasto de consumo per capita de los hogares y las ISFLSH',
                        'description': 'Es el valor de mercado de todos los bienes y servicios, incluidos los productos duraderos (como automoviles, lavadoras y computadoras domesticas), adquiridos por los hogares. Excluye las compras de viviendas, pero incluye el alquiler imputado de las viviendas ocupadas por sus propietarios.',
                        'mexico_info': 'En Mexico, el gasto ha ido incrementando gradualmente, desde poco mas de $6400 en el ano 2000 a $7400 en el ano 2023.',
                        'image': 'https://www.ine.gob.bo/wp-content/uploads/2019/11/p13-300x196.png',
                        'uom':'Dolares (2015)'
                    }
            ]

        ''' 
        self.available_features = [
            'CO2 Emissions (per capita)',
            'Control of Corruption',
            'Individuals using the Internet (% of population)',
            'Employees in industry (% of population)',
            'Gasto de consumo per capita de los hogares y las ISFLSH (US$ constantes de 2015)',
        ]
        '''
        
        self.model = None
        self.coef_ = None

    def fit(self, selected_features):
        """Fit the model based on selected features."""
        X = self.df_cleaned[selected_features]
        X_with_constant = sm.add_constant(X)
        
        # Fit the model using statsmodels
        self.model = sm.OLS(self.y, X_with_constant).fit()
        
        # Extract coefficients
        self.coef_ = {selected_features[i]: self.model.params[i+1] for i in range(len(selected_features))}
        self.coef_['Intercept'] = self.model.params[0]  # Add intercept term

    def get_model_info(self):
        """Return the coefficients for the model."""
        return self.coef_# if self.coef_ else {}

    def predict(self, X_new):
        """Make a prediction based on new input data."""
        X_new_with_const = sm.add_constant(X_new)
        return self.model.predict(X_new_with_const)

class Plotter:
    def __init__(self, coef):
        self.coef = coef

    @staticmethod
    def plot_actual_vs_predicted(y_test, y_pred):
        trace = go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            marker=dict(color='blue', size=8, opacity=0.7),
            text=[f"Actual: {actual:.2f}<br>Predicted: {pred:.2f}" for actual, pred in zip(y_test, y_pred)],
            hoverinfo='text'
        )
        layout = go.Layout(
            title='Actual vs Predicted Electrical Energy Consumption',
            xaxis=dict(title='Actual Electrical Energy Consumption (kWh per capita)'),
            yaxis=dict(title='Predicted Electrical Energy Consumption (kWh per capita)'),
            height=800, width=800, showlegend=False
        )
        fig = go.Figure(data=[trace], layout=layout)
        pyo.plot(fig, filename='templates/actual_vs_predicted.html')

    def predict(self, selected_variables, values):
        prediction = self.coef['Intercept']  # Start with the intercept
        for var_name, value in zip(selected_variables, values):  # Iterate through variables and their corresponding values
            prediction += self.coef.get(var_name, 0) * value  # Multiply coefficient by the value
        return prediction

    def update_plot(self, selected_variables, filepath):
        # Load the merged data
        df = pd.read_excel('Database2.xlsx')

        # Drop rows with missing values (NaN)
        df_cleaned = df.dropna()

        # Set the dependent variable (Y) - Use of electrical energy
        y = df_cleaned['Consumo de energia electrica (kWh per capita)']

        # Set the independent variables (X) - All other columns
        X = df_cleaned[selected_variables]

        # Add a constant to the X matrix (for the intercept)
        X_with_constant = sm.add_constant(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_with_constant, y, test_size=0.2, random_state=42)

        # Fit the model using statsmodels
        sm_model = sm.OLS(y_train, X_train).fit()

        # Predict the use of electrical energy on the test set
        y_pred = sm_model.predict(X_test)
        self.plot_actual_vs_predicted(y_test, y_pred)