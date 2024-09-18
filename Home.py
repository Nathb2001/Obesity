import streamlit as st
import pyAgrum as gum
import pyAgrum.causal as csl
import pandas as pd
import time
from datetime import date
from streamlit_gsheets import GSheetsConnection

st.set_page_config(
    page_title="Diagnosis",
    page_icon=":hamburger:",
)

conn = st.connection("gsheets", type=GSheetsConnection)
def update_database(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, ObesityLevel):
    conn = st.connection("gsheets", type=GSheetsConnection)
    datee=date.today()
    old_data=conn.read(worksheet="Sheet1",ttl=0)
    data = [[datee, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, ObesityLevel]]  # Ensure this is a list of lists
    info = pd.DataFrame(data, columns=['Date', 'Age', 'Height (CM)', 'Weight (KG)', 
    'How Many Main Meals in a Day', 'Eat Vegetables during each Meal', 
    'Eat Any Food between Meals', 'Eat High Caloric Food Frequently', 
    'Monitor Daily Calories', 'Family History Overweight', 
    'How Often Physical Activity', 'How much time do you use technological devices', 
    'Do you Drink Alcohol', 'Obesity Level'])    
    concat_data = pd.concat([old_data, info], ignore_index=True)
    conn.update(data=concat_data) #all this code is for updating db with new information, uses dataframe
    

def network(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12):
    bn = gum.BayesNet('Obesity')

    # Define variables and structure
    age = bn.add(gum.LabelizedVariable('Age', 'Age', ['Under 18','18 to 24', '25 to 34', '35 to 44', '45 to 54','55 to 65','65 and over']))
    height = bn.add(gum.LabelizedVariable('Height (CM)', 'Height (CM)', ['150 to 159','160 to 169','170 to 179','180 and over']))
    weight = bn.add(gum.LabelizedVariable('Weight (KG)', 'Weight (KG)', ['Under 50','50 to 59','60 to 69','70 to 79','80 to 89','90 to 99','Over 100']))
    familyHistory = bn.add(gum.LabelizedVariable('Family History Overweight', 'Family History Overweight', ['No','Yes']))
    highCalorieFoods = bn.add(gum.LabelizedVariable('Eat High Caloric Food Frequently', 'Eat High Caloric Food Frequently', ['No','Yes']))
    vegatables = bn.add(gum.LabelizedVariable('Eat Vegatables during each Meal', 'Eat Vegatables during each Meal', ['1','2','3']))
    mainMealsADay = bn.add(gum.LabelizedVariable('How Many Main Meals in a Day', 'How Many Main Meals in a Day', ['1','2','3','4']))
    foodBetweenMeals = bn.add(gum.LabelizedVariable('Eat Any Food between Meals', 'Eat Any Food between Meals', ['Rarely to Sometimes','Frequently to Alot']))
    monitor = bn.add(gum.LabelizedVariable('Monitor Daily Calories', 'Monitor Daily Calories', ['No','Yes']))
    physicalActivity = bn.add(gum.LabelizedVariable('How Often Physical Activity','How Often Physical Activity', ['No Physical Activity','Some Physical Activity','Good Amount of Physical Activity']))
    tech = bn.add(gum.LabelizedVariable('How much time do you use technological devices', 'How much time do you use technological devices', ['Less than hour','1 to 2 hours','2 to 4 hours','4 to 6 hours','6 to 8 hours','More than 8 hours']))
    alcohol = bn.add(gum.LabelizedVariable('Do you Drink Alcohol', 'Do you Drink Alcohol', ['No','Yes']))
    obesity = bn.add(gum.LabelizedVariable('Obesity Level', 'Obesity Level', ['Normal','Overweight','Obese']))

    links = [
        (alcohol, weight),
        (weight, obesity),
        (familyHistory, obesity),
        (physicalActivity, obesity),
        (physicalActivity, weight),
        (mainMealsADay, weight),
        (highCalorieFoods, weight),
        (tech, weight),
        (vegatables, weight),
        (height, weight),
        (monitor, highCalorieFoods),
        (foodBetweenMeals, mainMealsADay),
        (age, height),
        (age, weight),
    ]

    for link in links:
        bn.addArc(*link)

    df = pd.read_csv('Obesity_data.csv')
    learner = gum.BNLearner(df, bn)
    learner.useEM(1e-3)
    learner.useSmoothingPrior()
    bn = learner.learnParameters(bn.dag())

    ie = gum.LazyPropagation(bn)
    ie.makeInference()
    ie.setEvidence({
        'Age': Q1,
        'Height (CM)': Q2,
        'Weight (KG)': Q3,
        'How Many Main Meals in a Day': Q4,
        'Eat Vegatables during each Meal': Q5,
        'Eat Any Food between Meals': Q6,
        'Eat High Caloric Food Frequently': Q7,
        'Monitor Daily Calories': Q8,
        'Family History Overweight': Q9,
        'How Often Physical Activity': Q10,
        'How much time do you use technological devices': Q11,
        'Do you Drink Alcohol': Q12,
    })
    ie.makeInference()

    Normal = ie.posterior(obesity)[0]
    Overweight = ie.posterior(obesity)[1]
    Obese = ie.posterior(obesity)[2]

    obesityLevel, colour = obesity_level(Normal, Overweight, Obese)
    st.session_state.obesityLevel = obesityLevel
    st.session_state.colour = colour 

    st.session_state.bn = bn
    return obesityLevel, colour, bn

def obesity_level(Normal, Overweight, Obese):
    if Normal > Overweight and Normal > Obese:
        return 'Normal', ':green'
    elif Overweight > Normal and Overweight > Obese:
        return 'Overweight', ':orange'
    elif Obese > Normal and Obese > Overweight:
        return 'Obese', ':red'
    else:
        st.write('Unfortunately the lack of data for this case means there is an equal chance of being Normal, Overweight, or Obese.')
def counterfactual(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, option, option1):
    bn = st.session_state.bn
    obesityLevel = st.session_state.obesityLevel
    cm = csl.CausalModel(bn)
    pot = csl.counterfactual(cm=cm,
                             profile={
                                 'Age': Q1,
                                 'Height (CM)': Q2,
                                 'Weight (KG)': Q3,
                                 'How Many Main Meals in a Day': Q4,
                                 'Eat Vegatables during each Meal': Q5,
                                 'Eat Any Food between Meals': Q6,
                                 'Eat High Caloric Food Frequently': Q7,
                                 'Monitor Daily Calories': Q8,
                                 'Family History Overweight': Q9,
                                 'How Often Physical Activity': Q10,
                                 'How much time do you use technological devices': Q11,
                                 'Do you Drink Alcohol': Q12,
                                 'Obesity Level': obesityLevel
                             }, whatif={option},
                             on={'Obesity Level'},
                             values={option: option1})
    Normal = pot[0]
    Overweight = pot[1]
    Obese = pot[2]
    newObesity, colour = obesity_level(Normal, Overweight, Obese)
    st.write(f'If you changed {option} to {option1}, the predicted obesity level would be {colour}[**{newObesity}**].')

# Initialize session state variables only if they don't exist
if 'done' not in st.session_state:
    st.session_state.done = 0
if 'obesityLevel' not in st.session_state:
    st.session_state.obesityLevel = None
if 'bn' not in st.session_state:
    st.session_state.bn = None

# User interface
st.title('Obesity Prediction Web App :hamburger:')
st.markdown('This Web App uses **probability** established from previous data using different factors to diagnose Obesity.')
st.markdown('Try it out by answering 12 quick questions below:')

# Collect inputs from the user
Q1 = st.selectbox("**Question 1: What is your Age range?**", ['Under 18', '18 to 24', '25 to 34', '35 to 44', '45 to 54', '55 to 65', '65 and over'])
Q2 = st.selectbox("**Question 2: What is your Height range (in CM)?**", ['150 to 159', '160 to 169', '170 to 179', '180 and over'])
Q3 = st.selectbox("**Question 3: What is your Weight range (in KG)?**", ['Under 50', '50 to 59', '60 to 69', '70 to 79', '80 to 89', '90 to 99', 'Over 100'])
Q4 = st.selectbox("**Question 4: How many main meals do you have in a day?**", ['1', '2', '3', '4'])
Q5 = st.selectbox("**Question 5: How many vegetables do you eat during each meal?**", ['1', '2', '3'])
Q6 = st.selectbox("**Question 6: How often do you eat food between meals?**", ['Rarely to Sometimes', 'Frequently to Alot'])
Q7 = st.selectbox("**Question 7: Do you eat high caloric food frequently?**", ['Yes', 'No'])
Q8 = st.selectbox("**Question 8: Do you monitor your daily calorie intake?**", ['Yes', 'No'])
Q9 = st.selectbox("**Question 9: Does your family have a history of being overweight?**", ['Yes', 'No'])
Q10 = st.selectbox("**Question 10: How often do you engage in physical activity?**", ['No Physical Activity', 'Some Physical Activity', 'Good Amount of Physical Activity'])
Q11 = st.selectbox("**Question 11: How much time do you spend using technological devices?**", ['Less than hour', '1 to 2 hours', '2 to 4 hours', '4 to 6 hours', '6 to 8 hours', 'More than 8 hours'])
Q12 = st.selectbox("**Question 12: Do you drink alcohol?**", ['Yes', 'No'])

if st.button("Submit"):
    progress_text = "Analysing Data"
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.03)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    obesityLevel, colour, bn = network(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12)
    st.session_state.done = 1  # Mark as done after prediction
    update_database(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, obesityLevel)

# Show counterfactual analysis if prediction is done
try:
    if st.session_state.done == 1:
        obesityLevel, colour, bn = network(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12)
        if 'obesityLevel' in st.session_state and st.session_state.obesityLevel is not None:
            st.write(f'The prediction of Obesity Level based on the factors given is {st.session_state.colour}[**{st.session_state.obesityLevel}**].')
        st.header('Counterfactual Prediction Section :brain:')
        st.markdown('Here you can see if your **Obesity Level** would have been **different** if you made a change to your lifestyle:')
        option = st.selectbox('What would you like to **change**?', ['Age', 'Weight (KG)', 'How Many Main Meals in a Day', 'Eat Vegatables during each Meal', 'Eat Any Food between Meals', 'Eat High Caloric Food Frequently', 'Monitor Daily Calories', 'Family History Overweight', 'How Often Physical Activity', 'How much time do you use technological devices', 'Do you Drink Alcohol'])

        if option == 'Age':
            option1 = st.selectbox('What would you like to change Age to?', ['Under 18', '18 to 24', '25 to 34', '35 to 44', '45 to 54', '55 to 65', '65 and over'])
        elif option == 'Weight (KG)':
            option1 = st.selectbox('What would you like to change Weight to?', ['Under 50', '50 to 59', '60 to 69', '70 to 79', '80 to 89', '90 to 99', 'Over 100'])
        elif option == 'How Many Main Meals in a Day':
            option1 = st.selectbox('What would you like to change Main Meals to?', ['1', '2', '3', '4'])
        elif option == 'Eat Vegatables during each Meal':
            option1 = st.selectbox('What would you like to change Vegatables to?', ['1', '2', '3'])
        elif option == 'Eat Any Food between Meals':
            option1 = st.selectbox('What would you like to change Food between Meals to?', ['Rarely to Sometimes', 'Frequently to Alot'])
        elif option == 'Eat High Caloric Food Frequently':
            option1 = st.selectbox('What would you like to change Eat High Caloric Food Frequently to?', ['Yes', 'No'])
        elif option == 'Monitor Daily Calories':
            option1 = st.selectbox('What would you like to change Monitor Daily Calories to?', ['Yes', 'No'])
        elif option == 'Family History Overweight':
            option1 = st.selectbox('What would you like to change Family History to?', ['Yes', 'No'])
        elif option == 'How Often Physical Activity':
            option1 = st.selectbox('What would you like to change Physical Activity to?', ['No Physical Activity', 'Some Physical Activity', 'Good Amount of Physical Activity'])
        elif option == 'How much time do you use technological devices':
            option1 = st.selectbox('What would you like to change Technology Time to?', ['Less than hour', '1 to 2 hours', '2 to 4 hours', '4 to 6 hours', '6 to 8 hours', 'More than 8 hours'])
        elif option == 'Do you Drink Alcohol':
            option1 = st.selectbox('What would you like to change Alcohol Consumption to?', ['Yes', 'No'])

        if st.button('Submit Change'):
            with st.spinner('Processing counterfactual...'):
                counterfactual(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, option, option1)
except TypeError as e:
    st.write('Unfortunately the lack of data for this case means we cannot run counterfactual modelling')    
