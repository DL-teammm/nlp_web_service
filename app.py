import os
import pandas as pd
import streamlit as st

def input_data(samples_path: str = 'samples_resume', user_path: str = 'user_resume') -> str:
    samples = [
        resume_name.replace('.csv', '') for resume_name in os.listdir(samples_path)
    ]

    row2_spacer1, row2_1, row2_spacer2 = st.columns((0.1, 3.2, 0.1))

    with row2_1:
        resume = st.selectbox(
            "Select one of our sample resume",
            tuple(samples),
            index=None,
            placeholder="Select sample resume...",
        )

        st.markdown("**or**")

        uploaded_file = st.file_uploader("Pick a resume")
        if uploaded_file is not None:
            save_uploaded_file(uploaded_file, user_path)
            resume = uploaded_file.name

        if uploaded_file is None:
            st.warning("""File should be in csv format""")

        st.write('You selected:', resume)

        if uploaded_file is None:
            resume_path = os.path.join(
                samples_path, f'{resume}.csv',
            )
        else:
            resume_path = pd.read_csv(f'./{user_path}/{uploaded_file.name}')

        return resume_path


def save_uploaded_file(uploaded_file, user_resume):
    with open(os.path.join(user_resume, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return st.success("Saved File: {} to user_resume".format(uploaded_file.name))


def show_chain_results():

    st.header('Recommended vacancies:')

    df = pd.read_csv("./results/vacancies_example.csv")
    header = [
        "vacancy_name",
        "full_company_name",
        "regionName",
        "salary",
        "schedule_type",
        "position_requirements",
        "position_responsibilities",
        ]
    df.to_csv('./results/filtered_vacancies.csv', columns=header, index=False)

    new_df = pd.read_csv("./results/filtered_vacancies.csv")

    st.write(new_df)

def build_page():
    st.set_page_config(layout="wide")
    st.title('Vacancy Recommendation Project')

    row1_spacer1, row1_1, row1_spacer2 = st.columns((0.1, 3.2, 0.1))

    with row1_1:
        st.markdown(
            "Service without model"
        )
        st.markdown(
            ""
        )

    input_data()

    show_chain_results()

if __name__ == '__main__':
    build_page()