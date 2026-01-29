import numpy as np
import joblib 
import streamlit  as st

model_cluster = joblib.load('./models/model_cluster.pkl')
model = joblib.load('./models/model.pkl')
scale_cluster = joblib.load('./models/scale_cluster.pkl')
scale_model = joblib.load('./models/scale_model.pkl')



st.title('Leadership Model')

manager_role = st.selectbox("Manager Role",['HR Manager','Team Lead', 'Project Manager',\
       'Operations Manager','Department Head'])
team_size = st.number_input("Team Size", min_value=1, max_value=100, value=5)
workload_ui = st.slider("Workload", 1, 10, 5)
workload = workload_ui / 10

communication_ui = st.slider("Communication", 1, 10, 5)
communication = communication_ui / 10

support_ui = st.slider("Support", 1, 10, 5)
support = support_ui / 10

learning_culture_ui = st.slider("Learning Culture", 1, 10, 5)
learning_culture = learning_culture_ui / 10

reward_fairness_ui = st.slider("Reward Fairness", 1, 10, 5)
reward_fairness = reward_fairness_ui / 10

decision_speed_ui = st.slider("Decision Speed", 1, 10, 5)
decision_speed = decision_speed_ui / 10

stress_ui = st.slider("Stress Level", 1, 10, 5)
stress = stress_ui / 10

burnout_ui = st.slider("Burnout Level", 1, 10, 5)
burnout = burnout_ui / 10

turnover_intent_ui = st.slider("Turnover Intent", 1, 10, 5)
turnover_intent = turnover_intent_ui / 10



cluster_insights = {
    0: {
        "name": "Support-Driven, Low-Stress Environment",
        "characteristics": [
            "Small team sizes",
            "High levels of communication",
            "Strong managerial support",
            "Low workload pressure",
            "Low stress and burnout",
            "Moderate to high employee engagement"
        ]
    },

    1: {
        "name": "High-Pressure, Workload-Intensive Environment",
        "characteristics": [
            "Very high workload",
            "High stress levels",
            "High burnout",
            "Limited communication",
            "Limited support mechanisms",
            "Low employee engagement"
        ]
    },

    2: {
        "name": "Highly Supportive, High-Engagement Environment",
        "characteristics": [
            "Very high communication",
            "Very high managerial support",
            "High reward fairness",
            "Very low stress and burnout",
            "Stable work environment",
            "Highest employee engagement"
        ]
    },

    3: {
        "name": "Large-Scale, Operationally Intensive Environment",
        "characteristics": [
            "Large team sizes",
            "High operational workload",
            "High stress and burnout",
            "Moderate communication levels",
            "Slower decision-making processes",
            "Low to moderate employee engagement"
        ]
    }
}






if st.button('Submit'):

       role_hr = 1 if manager_role == 'HR Manager' else 0
       role_ops = 1 if manager_role == 'Operations Manager' else 0
       role_pm = 1 if manager_role == 'Project Manager' else 0
       role_tl = 1 if manager_role == 'Team Lead' else 0
       role_dh=1 if manager_role == 'Department Head' else 0

       x_cluster =  np.array([[team_size,workload,communication,support,learning_culture,reward_fairness,decision_speed,\
                         stress,burnout,turnover_intent,role_dh,role_hr,role_ops,role_pm,role_tl]])

       x_cluster_scaled = scale_cluster.transform(x_cluster)
       cluster = model_cluster.predict(x_cluster_scaled)[0]
       
       st.subheader('Cluster')
       st.success(f"Employee belongs to Cluster: {cluster}")

       info = cluster_insights[cluster]

       st.subheader(info["name"])

       st.markdown("### 🔍 Cluster Characteristics")
       for c in info["characteristics"]:
              st.write("•", c)       

       x_model =  np.array([[team_size,workload,communication,support,learning_culture,reward_fairness,decision_speed,\
                         stress,burnout,turnover_intent,role_dh,role_hr,role_ops,role_pm,role_tl,cluster]])

       x_model_scaled = scale_model.transform(x_model)

       engagement_score = model.predict(x_model_scaled)[0]

       st.subheader("Engagement Score")
       st.success(f"Predicted Engagement Score: {engagement_score:.2f}")



