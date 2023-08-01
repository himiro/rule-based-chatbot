import streamlit as st
from streamlit_chat import message


def run_streamlit_app(chatbot, data):
    # Creating the chatbot interface
    st.title("Rule-based Chatbot")

    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    # clear input
    def clear_input():
        st.session_state.user_input = st.session_state.input
        st.session_state.input = ''

    # We will get the user's input by calling the get_text function
    def get_text():
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ''

        st.text_input("You: ",
                      "",
                      placeholder='Ask your question',
                      key="input",
                      on_change=clear_input)

        return st.session_state.user_input

    user_input = get_text()

    if user_input:
        output = chatbot.generate_response(user_input, data)

        # store the output
        st.session_state.generated.append(output)
        st.session_state.past.append(user_input)

    if st.session_state['generated']:

        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            if i < len(st.session_state['generated']):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
