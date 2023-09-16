import React, { Component } from "react";
import axios from "axios"; 

//Importing from the Wolox Chat Widget
import {Widget as ChatWidget, 
        toggleWidget,
        addResponseMessage,
        toggleMsgLoader,
        toggleInputDisabled
    } from "react-chat-widget";

//CSS Styles
import 'react-chat-widget/lib/styles.css';
import "./Chatbox.css";

class Chatbox extends Component {
    
    //Setting up dynamically changing objects
    constructor(props) {
        super(props);
        this.state = {
          userMessages: [],
          chatbotMessages: []
        };
    }
    
    componentDidMount() {
        toggleWidget();
    }
    
    handleNewUserMessage = (newMessage) => {
        this.state.userMessages.push(newMessage); 

        console.log(`New message incomig! ${newMessage}`);
        
        toggleMsgLoader(); //Toggle on
        
        //This becomes a promise because of the promise in postChatbot        
        this.postChatbotMessage(newMessage)
            .then(chatbotResponse => {
                toggleMsgLoader(); //Toggle off
                addResponseMessage(chatbotResponse);
            });
        
    }

     postChatbotMessage = (messageInput) => {
        this.state.userMessages.push(messageInput); 

        let postUrl = "https://angel-hacks-2021-jcww4.ondigitalocean.app/api/message/" + messageInput;

        
        let chatbotResponse; 

        //Get route to receive chatbot response
        return axios.get(postUrl)
            .then(response => {
                chatbotResponse = response.data; 
                this.state.chatbotMessages.push(chatbotResponse); 
                return chatbotResponse; 
            });
    }

    //The component
    render() {
        return (
            <div>
                <ChatWidget 
                    showChat="true" 
                    fullScreenMode="true"
                    showCloseButton="false"
                    handleNewUserMessage={this.handleNewUserMessage}
                />
            </div>
        );
    }

}

export default Chatbox;